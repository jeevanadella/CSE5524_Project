import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model


class BioClip2_DeepFeatureRegressorWithDomainID(nn.Module):
    def __init__(
        self,
        bioclip,
        num_features=768,
        hidden_size_begin=512,
        hidden_layer_decrease_factor=4,
        num_outputs=3,
        n_last_trainable_resblocks=6,
        known_domain_ids=None,
        use_lora=True,
    ):
        super().__init__()
        # regressor linear layer
        self.n_last_trainable_resblocks = n_last_trainable_resblocks
        self.bioclip = bioclip
        self.known_domain_ids = known_domain_ids
        self.use_lora = use_lora
        
        # Apply LoRA to the last n_last_trainable_resblocks
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
        
        if known_domain_ids:
            self.padding_idx = len(known_domain_ids)
            self.domain_id_feature_extractor = nn.Sequential(
                nn.Embedding(
                    num_embeddings=len(known_domain_ids) + 1,
                    embedding_dim=num_features,
                    padding_idx=self.padding_idx,
                ),
                nn.Linear(in_features=num_features, out_features=num_features),
                nn.GELU(),
                nn.Linear(in_features=num_features, out_features=num_features),
                nn.LayerNorm(num_features),
            )
        else:
            self.padding_idx = 0
            self.known_domain_ids = []

        self.regressor = nn.Sequential(
            # 768 = num features output from bioclip
            nn.Linear(in_features=num_features, out_features=hidden_size_begin),
            nn.GELU(),
            nn.Linear(
                in_features=hidden_size_begin,
                out_features=int(hidden_size_begin / hidden_layer_decrease_factor),
            ),
            nn.GELU(),
            nn.Linear(
                in_features=int(hidden_size_begin / hidden_layer_decrease_factor),
                out_features=int(hidden_size_begin / hidden_layer_decrease_factor**2),
            ),
            nn.GELU(),
            nn.Linear(
                in_features=int(hidden_size_begin / hidden_layer_decrease_factor**2),
                out_features=num_outputs,
            ),
        )

    def get_trainable_parameters(self, lr=0.003):
        param_groups = []
        
        # Layer-wise learning rate decay for LoRA parameters
        resblocks = self.bioclip.visual.transformer.resblocks
        start_idx = len(resblocks) - self.n_last_trainable_resblocks
        
        for idx, block in enumerate(resblocks[-self.n_last_trainable_resblocks:]):
            layer_params = []
            if self.use_lora:
                for name, param in block.named_parameters():
                    if "lora" in name or "magnitude_vector" in name:
                        layer_params.append(param)
            else:
                layer_params = list(block.parameters())
            
            if layer_params:
                # Deeper layers (later in list) get higher LR
                layer_lr_multiplier = 1.0 + (idx / self.n_last_trainable_resblocks) * 0.5
                param_groups.append({
                    "params": layer_params,
                    "lr": lr * layer_lr_multiplier,
                })
        
        # LayerNorm gets moderate LR
        param_groups.append({
            "params": list(self.bioclip.visual.ln_post.parameters()),
            "lr": lr * 0.5,
        })
        
        # Regressor gets full LR  
        param_groups.append({
            "params": list(self.regressor.parameters()),
            "lr": lr * 1.5,
        })
        
        return param_groups

    def forward(self, x, domain_ids=None):
        h = self.forward_frozen(x)
        return self.forward_unfrozen(h, domain_ids)

    def forward_vision_transformer_before(self, x, attn_mask=None):
        if not self.bioclip.visual.transformer.batch_first:
            x = x.transpose(0, 1).contiguous()  # NLD -> LND

        for r in self.bioclip.visual.transformer.resblocks[
            : -self.n_last_trainable_resblocks
        ]:
            x = r(x, attn_mask=attn_mask)

        return x

    def forward_vision_transformer_after(self, x, attn_mask=None):
        for r in self.bioclip.visual.transformer.resblocks[
            -self.n_last_trainable_resblocks :
        ]:
            x = r(x, attn_mask=attn_mask)

        if not self.bioclip.visual.transformer.batch_first:
            x = x.transpose(0, 1)  # LND -> NLD
        return x

    def forward_frozen(self, x):
        x = self.bioclip.visual._embeds(x)
        x = self.forward_vision_transformer_before(x)

        return x

    def forward_unfrozen(self, x, domain_ids=None):
        x = self.forward_vision_transformer_after(x, attn_mask=None)

        pooled, tokens = self.bioclip.visual._pool(x)

        if self.bioclip.visual.proj is not None:
            pooled = pooled @ self.bioclip.visual.proj

        if self.bioclip.visual.output_tokens:
            features = pooled, tokens
        else:
            features = pooled
        features = F.normalize(features, dim=-1)

        if len(self.known_domain_ids) == 0:
            return self.regressor(features)

        if domain_ids is None:
            domain_id_features = (
                torch.ones(features.shape[0]).to(features.device) * self.padding_idx
            )
        else:
            domain_ids = torch.tensor(
                [
                    self.known_domain_ids.index(did)
                    if did in self.known_domain_ids
                    else self.padding_idx
                    for did in domain_ids
                ]
            ).to(features.device)
            domain_id_features = self.domain_id_feature_extractor(domain_ids)

        return self.regressor(features + domain_id_features)
