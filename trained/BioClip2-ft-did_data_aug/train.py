import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from datasets import load_dataset

from utils import (
    get_training_args,
    get_bioclip,
    evalute_spei_r2_scores,
    extract_deep_features_with_domain_id,
    get_collate_fn,
)
from model import BioClip2_DeepFeatureRegressorWithDomainID


def train(model, dataloader, val_dataloader, lr, epochs, domain_id_aug_prob, feature_noise_std, 
          specimen_dropout_prob, feature_mixup_alpha, progressive_aug, aug_warmup_pct, save_dir):
    optimizer = optim.Adam(model.get_trainable_parameters(lr=lr))
    loss_fn = nn.MSELoss()
    best_r2 = -1.0
    best_epoch = 0
    save_path = Path(save_dir, "model.pth")
    

    warmup_epochs = int(epochs * aug_warmup_pct)
    
    print("=" * 60)
    print("Training Configuration")
    print("=" * 60)
    print(f"Progressive Augmentation: {'ENABLED' if progressive_aug else 'DISABLED'}")
    if progressive_aug:
        print(f"Warmup Schedule: Linear ramp over first {warmup_epochs} epochs ({aug_warmup_pct*100:.0f}% of training)")
    print(f"Max augmentation settings:")
    print(f"  - Domain ID dropout: {domain_id_aug_prob:.2f}")
    print(f"  - Feature noise std: {feature_noise_std:.3f}")
    print(f"  - Specimen dropout: {specimen_dropout_prob:.2f}")
    print(f"  - Mixup alpha: {feature_mixup_alpha:.2f}")
    print("=" * 60)
    print("begin training")
    
    tbar = tqdm(range(epochs), position=0, leave=True)
    for epoch in tbar:

        if progressive_aug and epoch < warmup_epochs:
            aug_strength = epoch / warmup_epochs
        else:
            aug_strength = 1.0
        
        current_domain_aug_prob = aug_strength * domain_id_aug_prob
        current_noise_std = aug_strength * feature_noise_std
        current_dropout_prob = aug_strength * specimen_dropout_prob
        current_mixup_alpha = aug_strength * feature_mixup_alpha
        
        model.train()
        epoch_loss = 0
        inner_tbar = tqdm(dataloader, "training model", position=1, leave=False)
        preds = []
        gts = []
        for feats, y, did in inner_tbar:
            batch_size = feats.size(0)
            
            if current_dropout_prob > 0 and batch_size > 1:
                keep_mask = torch.rand(batch_size) > current_dropout_prob
                if keep_mask.sum() == 0:
                    keep_mask[0] = True
                feats = feats[keep_mask]
                y = y[keep_mask]
                did = [did[i] for i, keep in enumerate(keep_mask) if keep]
            
            if current_mixup_alpha > 0 and feats.size(0) > 1:
                lam = np.random.beta(current_mixup_alpha, current_mixup_alpha)
                indices = torch.randperm(feats.size(0))
                feats = lam * feats + (1 - lam) * feats[indices]
                y = lam * y + (1 - lam) * y[indices]
            
            if torch.rand(1).item() < current_domain_aug_prob:
                did = [model.padding_idx for _ in range(len(did))]
            
            if current_noise_std > 0:
                noise = torch.randn_like(feats) * current_noise_std
                feats = feats + noise
            
            y = y.cuda()
            optimizer.zero_grad()
            outputs = model.forward_unfrozen(feats.cuda(), domain_ids=did)
            loss = loss_fn(y, outputs)
            loss.backward()
            optimizer.step()

            epoch_loss = epoch_loss + loss
            preds.extend(outputs.detach().cpu().numpy().tolist())
            gts.extend(y.detach().cpu().numpy().tolist())
            inner_tbar.set_postfix({"loss": loss.item()})

        gts = np.array(gts)
        preds = np.array(preds)
        spei_30_r2, spei_1y_r2, spei_2y_r2 = evalute_spei_r2_scores(gts, preds)
        log_dict = {
            "train_loss": epoch_loss.item() / len(dataloader),
            "epoch": epoch,
            "train_SPEI_30d_r2": spei_30_r2,
            "train_SPEI_1y_r2": spei_1y_r2,
            "train_SPEI_2y_r2": spei_2y_r2,
        }
        
        if progressive_aug:
            log_dict["aug_strength"] = aug_strength
        
        tbar.set_postfix(log_dict)

        epoch_loss = 0
        inner_tbar = tqdm(val_dataloader, "validating model", position=1, leave=False)
        preds = []
        gts = []
        model.eval()
        with torch.no_grad():
            for feats, y, did in inner_tbar:
                y = y.cuda()
                outputs = model.forward_unfrozen(feats.cuda(), domain_ids=did)
                loss = loss_fn(y, outputs)

                epoch_loss = epoch_loss + loss
                preds.extend(outputs.detach().cpu().numpy().tolist())
                gts.extend(y.detach().cpu().numpy().tolist())
                inner_tbar.set_postfix({"loss": loss.item()})

        gts = np.array(gts)
        preds = np.array(preds)
        spei_30_r2, spei_1y_r2, spei_2y_r2 = evalute_spei_r2_scores(gts, preds)
        log_dict |= {
            "val_loss": epoch_loss.item() / len(dataloader),
            "val_SPEI_30d_r2": spei_30_r2,
            "val_SPEI_1y_r2": spei_1y_r2,
            "val_SPEI_2y_r2": spei_2y_r2,
        }

        avg_val_r2 = sum([spei_30_r2, spei_1y_r2, spei_2y_r2]) / 3.0
        if avg_val_r2 >= best_r2:
            best_r2 = avg_val_r2
            best_epoch = epoch
            
            torch.save(model.state_dict(), save_path)
        
        log_dict |= {
            "best_epoch": best_epoch,
            "best_val_r2": best_r2,
        }
        tbar.set_postfix(log_dict)

    model.load_state_dict(torch.load(save_path))
    print("DONE!")

def main():
    # Get training arguments
    args = get_training_args()
    
    # Get datasets
    ds = load_dataset(
        "imageomics/sentinel-beetles",
        token=args.hf_token,
    )
    
    known_domain_ids = list(set([x for x in ds["train"]["domainID"]]))
    save_dir = Path(__file__).resolve().parent
    with open(save_dir / "known_domain_ids.json", "w") as f:
        json.dump(known_domain_ids, f)
    
    # load bioclip and model
    bioclip, transforms = get_bioclip()
    model = BioClip2_DeepFeatureRegressorWithDomainID(bioclip, n_last_trainable_resblocks=args.n_last_trainable_blocks, known_domain_ids=known_domain_ids).cuda()
    
    # Transform images for model input
    def dset_transforms(examples):
        examples["pixel_values"] = [transforms(img.convert("RGB")) for img in examples["file_path"]]
        return examples
    
    train_dset = ds["train"].with_transform(dset_transforms)
    val_dset = ds["validation"].with_transform(dset_transforms)
    
    dataloaders = []
    for i, dset in enumerate([train_dset, val_dset]):

        dataloader = DataLoader(
            dataset=dset,
            batch_size=args.batch_size,
            shuffle=i == 0, # Shuffle only for training set
            num_workers=args.num_workers,
            collate_fn=get_collate_fn(["domainID"]),
        )


        # Extract features
        X, Y, DID = extract_deep_features_with_domain_id(dataloader, model)

        dataloader = DataLoader(
            dataset=torch.utils.data.TensorDataset(X, Y, DID),
            batch_size=args.batch_size,
            shuffle=i == 0, # Shuffle only for training set
            num_workers=args.num_workers,
        )
        dataloaders.append(dataloader)

    train_dataloader, val_dataloader = dataloaders

    # run model
    train(
        model=model,
        dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        lr=args.lr,
        epochs=args.epochs,
        domain_id_aug_prob=args.domain_id_aug_prob,
        feature_noise_std=args.feature_noise_std,
        specimen_dropout_prob=args.specimen_dropout_prob,
        feature_mixup_alpha=args.feature_mixup_alpha,
        progressive_aug=args.progressive_aug,
        aug_warmup_pct=args.aug_warmup_pct,
        save_dir=save_dir
    )



if __name__ == "__main__":
    main()
