"""
models/agrodynamicnet.py

Multi-branch feedforward architecture for sowing depth prediction.

Branches:
  - Soil branch: dense layers over soil physical/chemical features
  - Weather branch: dense layers over weather/climatology features
  - Crop+Location branch: embeddings for crop_name & state, concatenated
    with numeric crop/location features, through dense layers

Branches merge before the final prediction layers. Each branch can be
independently disabled (zeroed out) for the ablation study -- this is
the actual point of the multi-branch design, not decoration.

NOTE: weather features here are single climatology snapshots per
season, NOT a time sequence -- there is deliberately no LSTM/RNN here,
since that would misrepresent the data's structure.
"""

import torch
import torch.nn as nn


class AgroDynamicNet(nn.Module):
    def __init__(
        self,
        n_soil_features: int,
        n_weather_features: int,
        n_crop_numeric_features: int,
        n_crops: int,
        n_states: int,
        crop_embed_dim: int = 8,
        state_embed_dim: int = 4,
        branch_hidden: int = 32,
        merged_hidden: int = 64,
        dropout: float = 0.15,
    ):
        super().__init__()

        self.soil_branch = nn.Sequential(
            nn.Linear(n_soil_features, branch_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(branch_hidden),
            nn.Dropout(dropout),
            nn.Linear(branch_hidden, branch_hidden),
            nn.ReLU(),
        )

        self.weather_branch = nn.Sequential(
            nn.Linear(n_weather_features, branch_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(branch_hidden),
            nn.Dropout(dropout),
            nn.Linear(branch_hidden, branch_hidden),
            nn.ReLU(),
        )

        self.crop_embedding = nn.Embedding(n_crops, crop_embed_dim)
        self.state_embedding = nn.Embedding(n_states, state_embed_dim)
        crop_branch_input_dim = crop_embed_dim + state_embed_dim + n_crop_numeric_features
        self.crop_branch = nn.Sequential(
            nn.Linear(crop_branch_input_dim, branch_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(branch_hidden),
            nn.Dropout(dropout),
            nn.Linear(branch_hidden, branch_hidden),
            nn.ReLU(),
        )

        self.merged = nn.Sequential(
            nn.Linear(branch_hidden * 3, merged_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(merged_hidden, merged_hidden // 2),
            nn.ReLU(),
        )
        self.depth_head = nn.Linear(merged_hidden // 2, 1) 

    def forward(
        self,
        soil_x, weather_x, crop_numeric_x, crop_idx, state_idx,
        use_soil=True, use_weather=True, use_crop=True,
    ):
        """
        use_soil / use_weather / use_crop: set False to zero out that
        branch's contribution entirely -- this is what powers the
        ablation study (soil-only, weather-only, crop-only, combined).
        """
        soil_out = self.soil_branch(soil_x)
        if not use_soil:
            soil_out = torch.zeros_like(soil_out)

        weather_out = self.weather_branch(weather_x)
        if not use_weather:
            weather_out = torch.zeros_like(weather_out)

        crop_embed = self.crop_embedding(crop_idx)
        state_embed = self.state_embedding(state_idx)
        crop_input = torch.cat([crop_embed, state_embed, crop_numeric_x], dim=1)
        crop_out = self.crop_branch(crop_input)
        if not use_crop:
            crop_out = torch.zeros_like(crop_out)

        merged_input = torch.cat([soil_out, weather_out, crop_out], dim=1)
        merged_out = self.merged(merged_input)
        depth_pred = self.depth_head(merged_out)
        return depth_pred.squeeze(-1)