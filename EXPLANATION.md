# 🧠 How It Works — Deep Technical Explanation

This document explains every decision made in the fine-tuning pipeline, why each step exists, and what would break if it were removed.

---

## 1. Dataset Loading and Exploration

The fine-tuning pipeline begins by loading a dataset from HuggingFace Hub called `jackhhao/jailbreak-classification` using the `load_dataset` function. This dataset contains 1,306 examples split into train (1,044) and test (262) sets, where each example is a dictionary with two fields: `prompt` containing the raw text and `type` containing the label which is either `benign` or `jailbreak`. Before doing anything else, we explore the dataset by counting label distribution using `Counter`, finding 527 jailbreak and 517 benign examples — a roughly balanced dataset which is important because an imbalanced dataset would cause the model to learn a shortcut of always predicting the majority class rather than learning actual patterns.

---

## 2. Instruction Formatting

The raw dataset cannot be fed directly to the model because a language model doesn't understand a simple label like `jailbreak` or `benign` in isolation. It needs to see complete instruction-response pairs that teach it what job it's supposed to do. So every example is reformatted using a `format_prompt` function that takes the raw dictionary and returns a new dictionary with a single key called `text` containing a structured string with three sections: `### Instruction` which tells the model its role as a security system, `### Input` which contains the actual prompt to classify, and `### Response` which contains either `INJECTION ATTACK` or `SAFE` depending on the original label.

The `if label == "jailbreak"` condition here is not the model doing classification — it is simply preparing the training targets, converting dataset terminology into clear output text that the model will learn to produce. This function is applied to every row using `dataset.map()` which runs the function in parallel across multiple CPU cores and caches the results to disk, producing a new formatted dataset where each row now has the complete instruction template.

```
Raw example:
{
  "prompt": "Ignore all previous instructions...",
  "type": "jailbreak"
}

After format_prompt:
{
  "text": "### Instruction:
           You are a security system...
           ### Input:
           Ignore all previous instructions...
           ### Response:
           INJECTION ATTACK"
}
```

---

## 3. Token Length Filtering

Before loading the model, we check the token length of every formatted example because the model has a maximum sequence length of 512 tokens, meaning the entire formatted string including instruction, input, and response combined must fit within 512 tokens. The tokenizer converts each text string into a sequence of integer token IDs, and we measure the length of that sequence using `tokens['input_ids'].shape[1]`. We find that 211 examples exceed 512 tokens, which would cause a shape mismatch crash during training because Unsloth would try to truncate the input but the input and target tensors would end up with different sizes. These 211 examples are removed using `dataset.filter()`, leaving 833 clean training examples.

---

## 4. Model Loading with QLoRA

The model is then loaded using Unsloth's `FastLanguageModel.from_pretrained()` with `load_in_4bit=True`. This is **QLoRA** — Quantization combined with LoRA. Without quantization, loading Llama-3.2-3B's 3.2 billion parameters in standard 32-bit floating point would require approximately 12GB of GPU VRAM, which exceeds the 15GB available on a free Colab T4 GPU when accounting for activations and gradients during training.

Unsloth calls the `bitsandbytes` library internally to convert every weight from a 32-bit float into a 4-bit integer, reducing memory consumption by 8x and fitting the model comfortably within 4GB of VRAM. The frozen Llama weights are stored in 4-bit format to save memory, while all actual computations during training — matrix multiplications, gradient calculations, and LoRA weight updates — happen in 16-bit precision using either fp16 or bf16 depending on what the GPU supports. Both happen simultaneously and neither replaces the other.

The tokenizer is also loaded alongside the model and serves as the bridge between text and numbers — it splits input text into subword units called tokens and maps each token to a unique integer ID from a vocabulary of 128,256 tokens in Llama's case. The tokenizer remains completely unchanged throughout the entire fine-tuning process since we are only changing the model's behavior, not its vocabulary.

---

## 5. LoRA Adapter Attachment

After loading the base model, LoRA adapters are attached using `FastLanguageModel.get_peft_model()` from the PEFT library. **LoRA — Low Rank Adaptation** — works by freezing all of the original 3.2 billion Llama parameters so they cannot be updated during training, and instead injecting two small trainable matrices called `lora_A` and `lora_B` alongside seven specific projection layers in every attention and feedforward block.

### Target Modules

| Module | Location | Purpose |
|--------|----------|---------|
| `q_proj` | Attention | Query — what am I looking for? |
| `k_proj` | Attention | Key — what do I contain? |
| `v_proj` | Attention | Value — what will I send? |
| `o_proj` | Attention | Mixes all attention heads together |
| `gate_proj` | FFN | Controls which information passes through |
| `up_proj` | FFN | Expands to 4x dimension |
| `down_proj` | FFN | Compresses back to original dimension |

### How LoRA Works Mathematically

For a projection layer of shape (3072 × 3072) which contains 9.4 million parameters, LoRA instead creates `lora_A` of shape (3072 × 16) and `lora_B` of shape (16 × 3072), totaling only 98,304 parameters:

```
Original layer update:   ΔW of shape (3072 × 3072) = 9,437,184 params
LoRA approximation:      lora_B × lora_A            = 98,304 params

lora_A: (3072 → 16)  compress to bottleneck
lora_B: (16 → 3072)  expand back out

Forward pass:
output = frozen_layer(x) + lora_B(lora_A(x)) × (alpha/r)
         ↑ unchanged       ↑ learned adjustment
```

The number 16 is the rank `r` which controls the bottleneck size — information from the 3072-dimensional input is compressed down to 16 dimensions by `lora_A` and then expanded back to 3072 dimensions by `lora_B`. The product of these two small matrices effectively approximates a large weight update matrix without storing it explicitly, which is called **low-rank decomposition**.

`lora_B` is initialized to all zeros so that at the start of training the LoRA contribution is exactly zero and the model behaves identically to the original Llama, preventing chaotic updates before the model has learned any direction. Across all 28 layers of Llama-3.2-3B with 7 target modules each, the total trainable parameters come to 24 million out of 3.2 billion total — just **0.75%** — while the remaining 99.25% stay completely frozen.

---

## 6. Training Configuration

Training is managed by TRL's `SFTTrainer` — Supervised Fine-Tuning Trainer — which handles the entire training loop internally.

### Key Arguments Explained

**`dataset_text_field="text"`**
Tells the trainer to use the `text` column from the formatted dataset which contains the complete instruction-input-response string. If set to `prompt` instead, the trainer would only see the raw input with no instruction format and no response — the model could not learn.

**`per_device_train_batch_size=4` + `gradient_accumulation_steps=4`**
The GPU can only hold 4 examples at once. However, 4 examples produce a noisy gradient estimate. Gradient accumulation solves this by computing gradients across 4 consecutive batches and summing them before applying any weight update, giving an effective batch size of 16 examples per update with no extra memory cost:

```
Batch 1: compute gradients → store, don't update
Batch 2: compute gradients → add to stored
Batch 3: compute gradients → add to stored
Batch 4: compute gradients → NOW update weights with sum
                             zero gradients, repeat
```

**`warmup_steps=10`**
The learning rate does not start at its full value immediately. At step 1 the LoRA matrices are freshly initialized and the model has no stable sense of direction yet — a full-speed update would cause destructive weight changes. Warmup ramps the learning rate linearly from 0 to `2e-4` over the first 10 steps, letting the model stabilize before training at full speed:

```
Steps 1-10:   lr: 0 → 2e-4  (warmup)
Steps 11-100: lr: 2e-4       (constant)
```

**`fp16` / `bf16`**
Instead of 32-bit floats for all computations, 16-bit formats cut memory usage and speed up GPU matrix multiplications. `bf16` is preferred on modern GPUs for its wider numerical range and training stability. `fp16` is the fallback on older GPUs like the T4. The code auto-detects GPU capability using `torch.cuda.is_bf16_supported()`.

**`learning_rate=2e-4`**
Controls the size of each weight update: `new_weight = old_weight - (lr × gradient)`. Too large causes overshooting and unstable training. Too small learns too slowly. 2e-4 is the standard starting point for LoRA fine-tuning.

Training runs for exactly 100 gradient update steps taking approximately 12 minutes on a T4 GPU. The loss starts at 3.05 at step 10 when the model is essentially guessing randomly, and decreases to 2.00 by step 100 as the LoRA matrices gradually learn the pattern of classifying injection attacks versus safe prompts.

---

## 7. Inference

After training, the model is switched to inference mode using `FastLanguageModel.for_inference()` which disables dropout so every forward pass produces deterministic output. To classify a new prompt, the same instruction template used during training is reconstructed but this time without the response — the model must generate the response itself.

The text is tokenized into integer IDs and moved to the GPU using `.to(device)`. The `model.generate()` function then autoregressively produces new tokens one at a time: at each step it runs a full forward pass through all 28 transformer blocks, takes the logits of the last token position which are unnormalized scores over the entire 128,256-token vocabulary, and since `do_sample=False` uses greedy decoding which always selects the single highest-probability token. This repeats for up to `max_new_tokens=10` steps or until a stop token is generated.

The output token IDs are decoded back into text using `tokenizer.decode()` with `skip_special_tokens=True` which removes internal markers like `<|begin_of_text|>` and `<|end_of_text|>` that the tokenizer adds automatically. The response section is extracted by splitting on `### Response:` and taking only the first line to avoid repetition artifacts. A final string check normalizes any repetition like `INJECTION ATTACK ATTACK` into clean `INJECTION ATTACK`.

---

## 8. Evaluation

Evaluation is performed on the 202 test examples the model never saw during training. For each example, the true label from the dataset and the model's predicted label are collected into two lists `y_true` and `y_pred`. The predicted text is mapped back to dataset terminology: any output containing `INJECTION` is mapped to `jailbreak` and anything else is mapped to `benign`.

These two lists are passed to `sklearn.metrics.classification_report()` which computes:

- **Precision** — out of all times the model predicted injection, how often was it actually an injection
- **Recall** — out of all actual injections, how many did the model catch
- **F1 Score** — harmonic mean of precision and recall

```
Confusion Matrix:
              Predicted
              SAFE    ATTACK
Actual SAFE  [ 116      3 ]   ← 3 false alarms
       ATTACK[   2     81 ]   ← 2 missed attacks

116 → safe correctly identified   ✅
81  → attacks correctly caught    ✅
3   → false alarms                ❌ annoying but not dangerous
2   → missed real attacks         ❌ the dangerous failures
```

The final accuracy is 97.52% with an F1 score of 0.97, demonstrating that the 24 million trainable LoRA parameters successfully learned to distinguish prompt injection attack patterns from safe user inputs across completely unseen examples in just 100 training steps and 12 minutes on a free GPU.

---

## 9. Why This Approach Works

The key insight is that Llama-3.2-3B already understands language, context, and what injection attacks look like from its pre-training on vast internet text. Fine-tuning doesn't teach it new knowledge — it reshapes its output behavior to be consistent, fast, and formatted correctly for a specific classification task. LoRA makes this possible on consumer hardware by training only the adjustment matrices rather than the full model, and QLoRA makes it possible on a free GPU by compressing the frozen weights into 4-bit storage while keeping all computation in 16-bit precision.
