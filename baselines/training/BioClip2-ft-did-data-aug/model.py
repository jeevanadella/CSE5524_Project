import torch
import torch.nn as nn
import torch.nn.functional as F


class BioClip2_DeepFeatureRegressorWithDomainID(nn.Module):
    def __init__(
        self,
        bioclip,
        num_features=768,
        hidden_size_begin=512,
        hidden_layer_decrease_factor=4,
        num_outputs=3,
        n_last_trainable_resblocks=2,
        known_domain_ids=None,
    ):
        super().__init__()
        # regressor linear layer
        self.n_last_trainable_resblocks = n_last_trainable_resblocks
        self.bioclip = bioclip
        self.known_domain_ids = known_domain_ids
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
        feature_params = []
        for block in self.bioclip.visual.transformer.resblocks[
            -self.n_last_trainable_resblocks :
        ]:
            feature_params += list(block.parameters())

        feature_params += self.bioclip.visual.ln_post.parameters()
        return [
            {
                "params": feature_params,
                "lr": lr * 0.01,
            },
            {"params": list(self.regressor.parameters()), "lr": lr},
        ]

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
