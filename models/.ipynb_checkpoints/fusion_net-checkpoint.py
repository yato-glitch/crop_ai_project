import torch
import torch.nn as nn

class SpatialEncoder(nn.Module):
    """Lightweight 2D-CNN for satellite spatial feature extraction."""
    def __init__(self, in_channels: int = 5, embed_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, embed_dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class TemporalEncoder(nn.Module):
    """Processes 30-day temporal weather sequences using a Bi-LSTM[cite: 1]."""
    def __init__(self, input_size: int = 6, hidden_size: int = 32, embed_dim: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, (hn, _) = self.lstm(x)
        concat_hidden = torch.cat((hn[-2], hn[-1]), dim=1)
        return self.fc(concat_hidden)

class ContextEncoder(nn.Module):
    """Encodes tabular soil properties and metadata[cite: 1]."""
    def __init__(self, input_dim: int = 12, embed_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, embed_dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class MultimodalCropAI(nn.Module):
    """Fuses spatial, temporal, and context embeddings[cite: 1]."""
    def __init__(self, spatial_dim: int = 128, temporal_dim: int = 64, context_dim: int = 32):
        super().__init__()
        self.spatial_enc = SpatialEncoder(embed_dim=spatial_dim)
        self.temporal_enc = TemporalEncoder(embed_dim=temporal_dim)
        self.context_enc = ContextEncoder(embed_dim=context_dim)

        fusion_dim = spatial_dim + temporal_dim + context_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.yield_head = nn.Linear(64, 1)      # Yield regression head[cite: 1]
        self.stress_head = nn.Linear(64, 3)     # 3-class stress classification head[cite: 1]

    def forward(self, img: torch.Tensor, seq: torch.Tensor, tab: torch.Tensor):
        e_s = self.spatial_enc(img)
        e_t = self.temporal_enc(seq)
        e_c = self.context_enc(tab)

        fused = torch.cat([e_s, e_t, e_c], dim=1)
        latent = self.fusion(fused)

        yield_pred = self.yield_head(latent)
        stress_pred = self.stress_head(latent)

        return yield_pred, stress_pred