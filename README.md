# CSE5524 Project — HDR Scientific Mood Challenge (Beetles)

This repository contains our training, LoRA-enhanced model, and runnable scripts for the Imageomics HDR Scientific Mood Challenge: Beetles as Sentinel Taxa. It focuses on a BioCLIP2 fine-tuning setup with Domain-ID features, plus an advanced Low-Rank Adaptation (LoRA) modification applied to the last ViT blocks.

## 1) Installation

- Create and activate the course Conda environment as explained by the TA (use the exact Python/Torch CUDA versions provided by the TA).
- Additionally install the PEFT library required for LoRA:

```bash
pip install peft
```

Notes:
- The project’s base Python dependencies are listed in `requirements.txt`; if needed, you can install them with `pip install -r requirements.txt` after activating the Conda environment.

## 2) Advanced Algorithm (LoRA on BioCLIP2-ft-did)

Our advanced algorithm augments the baseline BioCLIP2-ft-did approach by adding LoRA adapters to the last N transformer blocks of BioCLIP2’s visual ViT. We only update adapter parameters (plus `ln_post` and the regression head), achieving parameter-efficient fine-tuning.

Key LoRA code changes (added on top of the original BioCLIP2-ft-did):

### a) Configure LoRA and wrap the last ViT blocks

```python
from peft import LoraConfig, get_peft_model

# Inside BioClip2_DeepFeatureRegressorWithDomainID.__init__(...)
self.use_lora = use_lora
if self.use_lora:
  peft_config = LoraConfig(
    r=256,
    lora_alpha=256,
    target_modules=["c_fc", "c_proj", "out_proj", "q_proj", "k_proj", "v_proj"],
    lora_dropout=0.0,
    bias="none",
    task_type=None,
    use_rslora=True,
    use_dora=True,
  )

  resblocks = self.bioclip.visual.transformer.resblocks
  start = len(resblocks) - self.n_last_trainable_resblocks
  for idx in range(start, len(resblocks)):
    resblocks[idx] = get_peft_model(resblocks[idx], peft_config)
```

### b) Train only LoRA adapter params (+ ln_post + regressor) with layer-wise LR scaling

```python
def get_trainable_parameters(self, lr=0.003):
  param_groups = []

  # LoRA/DoRA parameters from the last N blocks
  resblocks = self.bioclip.visual.transformer.resblocks
  for idx, block in enumerate(resblocks[-self.n_last_trainable_resblocks:]):
    layer_params = []
    if self.use_lora:
      for name, param in block.named_parameters():
        if "lora" in name or "magnitude_vector" in name:
          layer_params.append(param)
    else:
      layer_params = list(block.parameters())

    if layer_params:
      # Slightly higher LR for deeper layers
      layer_lr_multiplier = 1.0 + (idx / self.n_last_trainable_resblocks) * 0.5
      param_groups.append({
        "params": layer_params,
        "lr": lr * layer_lr_multiplier,
      })

  # LayerNorm (visual.ln_post) at moderate LR
  param_groups.append({
    "params": list(self.bioclip.visual.ln_post.parameters()),
    "lr": lr * 0.5,
  })

  # Regressor head at slightly higher LR
  param_groups.append({
    "params": list(self.regressor.parameters()),
    "lr": lr * 1.5,
  })

  return param_groups
```

### c) Instantiate the model with LoRA enabled and use frozen feature extraction

```python
model = BioClip2_DeepFeatureRegressorWithDomainID(
  bioclip,
  n_last_trainable_resblocks=args.n_last_trainable_blocks,
  known_domain_ids=known_domain_ids,
  use_lora=True,
).cuda()

# Precompute frozen features from the early ViT blocks, and then train only the unfrozen tail blocks + adapters
X, Y, DID = extract_deep_features_with_domain_id(dataloader, model)
outputs = model.forward_unfrozen(X.cuda(), domain_ids=DID)
```

These snippets correspond to the LoRA-modified implementation in our trained pipeline. See `trained/BioClip2-ft-did_lora/model.py`, `train.py`, and `utils.py` for the complete context.

## 3) Test/Validation Examples (sh + sbatch)

We provide four ready-to-run SLURM scripts you can submit via `sbatch` on your cluster:

- `bioclip2-ft-did_train.sh`: Baseline fine-tuning with Domain-ID features.
- `bioclip2-ft-did-data-aug_train.sh`: Baseline + data augmentation variant.
- `bioclip2-ft-did_combined_train.sh`: Combined training configuration.
- `bioclip2-ft-did_lora_train.sh`: LoRA-enabled fine-tuning on the last ViT blocks.

Submit any of them like this:

```bash
sbatch bioclip2-ft-did_train.sh
sbatch bioclip2-ft-did-data-aug_train.sh
sbatch bioclip2-ft-did_combined_train.sh
sbatch bioclip2-ft-did_lora_train.sh
```

## 4) Running & Outputs (Evaluation)

The training `.sh` scripts include an additional evaluation step that runs after training finishes. This computes losses and R² metrics for the three targets (SPEI_30d, SPEI_1y, SPEI_2y) and writes results alongside the trained model artifacts.
