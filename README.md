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

## 3) Progressive Augmentation (BioClip2-ft-did-data-aug)

Our progressive augmentation approach extends the baseline BioCLIP2-ft-did model by applying curriculum-based data augmentation during training. Instead of using maximum augmentation from the start, the augmentation intensity gradually increases over the first portion of training (warmup period), allowing the model to learn stable features before introducing more challenging perturbations.

Key augmentation techniques implemented:

### a) Four types of augmentation applied to extracted features

```python
def train(model, dataloader, val_dataloader, lr, epochs, domain_id_aug_prob, feature_noise_std, 
          specimen_dropout_prob, feature_mixup_alpha, progressive_aug, aug_warmup_pct, save_dir):
    
    warmup_epochs = int(epochs * aug_warmup_pct)
    
    for epoch in range(epochs):
        # Progressive augmentation: linearly ramp up intensity
        if progressive_aug and epoch < warmup_epochs:
            aug_strength = epoch / warmup_epochs
        else:
            aug_strength = 1.0
        
        current_domain_aug_prob = aug_strength * domain_id_aug_prob
        current_noise_std = aug_strength * feature_noise_std
        current_dropout_prob = aug_strength * specimen_dropout_prob
        current_mixup_alpha = aug_strength * feature_mixup_alpha
```

### b) Four augmentation strategies during training

```python
# 1. Specimen dropout: randomly drop specimens from batch
if current_dropout_prob > 0 and batch_size > 1:
    keep_mask = torch.rand(batch_size) > current_dropout_prob
    if keep_mask.sum() == 0:
        keep_mask[0] = True
    feats = feats[keep_mask]
    y = y[keep_mask]

# 2. Feature mixup: interpolate between specimen features
if current_mixup_alpha > 0 and feats.size(0) > 1:
    lam = np.random.beta(current_mixup_alpha, current_mixup_alpha)
    indices = torch.randperm(feats.size(0))
    feats = lam * feats + (1 - lam) * feats[indices]
    y = lam * y + (1 - lam) * y[indices]

# 3. Domain ID dropout: randomly mask domain information
if torch.rand(1).item() < current_domain_aug_prob:
    did = [model.padding_idx for _ in range(len(did))]

# 4. Feature noise: add Gaussian noise to features
if current_noise_std > 0:
    noise = torch.randn_like(feats) * current_noise_std
    feats = feats + noise
```

### c) Configuration parameters (adjustable via command line)

```python
parser.add_argument("--domain_id_aug_prob", type=float, default=0.35)
parser.add_argument("--feature_noise_std", type=float, default=0.02)
parser.add_argument("--specimen_dropout_prob", type=float, default=0.25)
parser.add_argument("--feature_mixup_alpha", type=float, default=0.2)
parser.add_argument("--aug_warmup_pct", type=float, default=0.25)  # 25% of epochs
parser.add_argument("--progressive_aug", action="store_true", default=True)
```

These augmentation techniques improve model generalization and robustness by:
- **Domain ID dropout (35%)**: Forces the model to learn features independent of domain-specific patterns
- **Feature noise (std=0.02)**: Increases feature-space robustness and prevents overfitting
- **Specimen dropout (25%)**: Simulates smaller batch sizes and reduces specimen-specific memorization
- **Feature mixup (α=0.2)**: Creates synthetic training samples by interpolating between specimens, encouraging smoother decision boundaries

The progressive augmentation schedule starts at 0% intensity and linearly ramps up to 100% over the first 25% of training epochs, allowing stable learning before full augmentation kicks in. See `trained/BioClip2-ft-did-data-aug/model.py`, `train.py`, and `utils.py` for the complete implementation.

## 4) Training Examples (sh + sbatch)

We provide four ready-to-run SLURM scripts you can submit via `sbatch` on your cluster. You may have to alter the conda envronment listed in the shell files if it does not match 'cse5524_env':

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

The training `.sh` scripts include an additional evaluation step that runs after training finishes. This computes losses and R² metrics for the three targets (SPEI_30d, SPEI_1y, SPEI_2y) and writes results alongside the trained model artifacts.

## 5) Evaluation/ Testing of Trained models

In the trained directory, there are three sub-directories each with a trained models of the data augmented, LoRA, and combined altered BioClip2 models.

To test these models using SLURM scripts you can submit via the `sbatch` on your cluster. You may have to alter the conda envronment listed in the shell files if it does not match 'cse5524_env':

Submit any of them like this: 

```bash
sbatch bioclip2-ft-did-data-aug_eval.sh
sbatch bioclip2-ft-did_combined_eval.sh
sbatch bioclip2-ft-did_lora_eval.sh
```

To run the python for each one run: 
python trained/BioClip2-ft-did-data-aug/evaluation.py

python trained/BioClip2-ft-did-combined/evaluation.py

python trained/BioClip2-ft-did-lora/evaluation.py

The files expect CUDA. This computes losses and R² metrics for the three targets (SPEI_30d, SPEI_1y, SPEI_2y) and writes results alongside the trained model artifacts
