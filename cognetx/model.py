import torch
import torch.nn as nn
import torchaudio
from torchvision import models
from typing import List, Dict, Any
from loguru import logger

# Configure loguru logger
logger.add(
    "model_debug.log",
    format="{time} {level} {message}",
    level="DEBUG",
)


# Speech Processing Module using Conformer
class SpeechEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        encoder_dim: int,
        num_layers: int,
        num_heads: int,
        depthwise_conv_kernel_size: int = 31,
    ):
        super(SpeechEncoder, self).__init__()
        logger.debug("Initializing SpeechEncoder")
        # Input projection layer to match encoder_dim
        self.input_proj = nn.Linear(input_dim, encoder_dim)
        self.conformer = torchaudio.models.Conformer(
            input_dim=encoder_dim,  # Updated to encoder_dim
            num_heads=num_heads,
            ffn_dim=encoder_dim,
            num_layers=num_layers,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, time_steps, feature_dim)
        logger.debug(f"SpeechEncoder input shape: {x.shape}")
        batch_size, time_steps, _ = x.shape

        # Create lengths tensor assuming all sequences are full length
        lengths = torch.full(
            (batch_size,),
            time_steps,
            dtype=torch.long,
            device=x.device,
        )
        logger.debug(f"SpeechEncoder lengths: {lengths}")

        # Project input to encoder_dim
        x = self.input_proj(x)
        logger.debug(
            f"SpeechEncoder projected input shape: {x.shape}"
        )

        # Pass through Conformer
        output, _ = self.conformer(x, lengths)
        # Output is (batch_size, time_steps, encoder_dim)

        # Pool over time dimension
        output = output.mean(dim=1)
        logger.debug(f"SpeechEncoder output shape: {output.shape}")
        return output  # (batch_size, encoder_dim)


# Vision Processing Module using CNN
class VisionEncoder(nn.Module):
    def __init__(self, output_dim: int):
        super(VisionEncoder, self).__init__()
        logger.debug("Initializing VisionEncoder")
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, 3, H, W)
        logger.debug(f"VisionEncoder input shape: {x.shape}")
        features = self.cnn(x)
        logger.debug(f"VisionEncoder output shape: {features.shape}")
        return features  # (batch_size, output_dim)


# Video Processing Module using 3D CNN
class VideoEncoder(nn.Module):
    def __init__(self, output_dim: int):
        super(VideoEncoder, self).__init__()
        logger.debug("Initializing VideoEncoder")
        self.cnn3d = models.video.r3d_18(pretrained=True)
        self.cnn3d.fc = nn.Linear(
            self.cnn3d.fc.in_features, output_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, 3, time_steps, H, W)
        logger.debug(f"VideoEncoder input shape: {x.shape}")
        features = self.cnn3d(x)
        logger.debug(f"VideoEncoder output shape: {features.shape}")
        return features  # (batch_size, output_dim)


# Multimodal Fusion Module
class FusionModule(nn.Module):
    def __init__(self, input_dims: List[int], hidden_dim: int):
        super(FusionModule, self).__init__()
        logger.debug("Initializing FusionModule")
        total_input_dim = sum(input_dims)
        self.fc = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self, features_list: List[torch.Tensor]
    ) -> torch.Tensor:
        # features_list: list of tensors [(batch_size, dim), ...]
        logger.debug(
            f"FusionModule input feature shapes: {[f.shape for f in features_list]}"
        )
        concatenated = torch.cat(features_list, dim=1)
        fused = self.fc(concatenated)
        logger.debug(f"FusionModule output shape: {fused.shape}")
        return fused  # (batch_size, hidden_dim)


# Text Generation Module using Transformer Decoder
class TextDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        decoder_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super(TextDecoder, self).__init__()
        logger.debug("Initializing TextDecoder")
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(
            embedding_dim, dropout
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=decoder_dim,
            dropout=dropout,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )
        self.fc_out = nn.Linear(embedding_dim, vocab_size)

    def forward(
        self, tgt: torch.Tensor, memory: torch.Tensor
    ) -> torch.Tensor:
        # tgt: (tgt_seq_len, batch_size)
        # memory: (batch_size, decoder_dim)
        logger.debug(f"TextDecoder tgt input shape: {tgt.shape}")
        logger.debug(
            f"TextDecoder memory input shape: {memory.shape}"
        )
        tgt_emb = self.embedding(
            tgt
        )  # (tgt_seq_len, batch_size, embedding_dim)
        tgt_emb = self.positional_encoding(tgt_emb)
        memory = memory.unsqueeze(0)  # (1, batch_size, decoder_dim)
        output = self.transformer_decoder(tgt_emb, memory)
        output = self.fc_out(output)
        logger.debug(f"TextDecoder output shape: {output.shape}")
        return output  # (tgt_seq_len, batch_size, vocab_size)


# Positional Encoding Module
class PositionalEncoding(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        dropout: float = 0.1,
        max_len: int = 5000,
    ):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        logger.debug("Initializing PositionalEncoding")

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2)
            * (-torch.log(torch.tensor(10000.0)) / embedding_dim)
        )
        pe = torch.zeros(max_len, 1, embedding_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


# Full Model
class CogNetX(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(CogNetX, self).__init__()
        logger.debug("Initializing MultimodalModel")
        self.speech_encoder = SpeechEncoder(
            input_dim=config["speech_input_dim"],
            encoder_dim=config["encoder_dim"],
            num_layers=config["speech_num_layers"],
            num_heads=config["speech_num_heads"],
            depthwise_conv_kernel_size=config[
                "depthwise_conv_kernel_size"
            ],
        )
        self.vision_encoder = VisionEncoder(
            output_dim=config["encoder_dim"]
        )
        self.video_encoder = VideoEncoder(
            output_dim=config["encoder_dim"]
        )
        self.fusion_module = FusionModule(
            input_dims=[config["encoder_dim"]] * 3,
            hidden_dim=config["decoder_dim"],
        )
        self.text_decoder = TextDecoder(
            vocab_size=config["vocab_size"],
            embedding_dim=config["embedding_dim"],
            decoder_dim=config["decoder_dim"],
            num_layers=config["decoder_num_layers"],
            num_heads=config["decoder_num_heads"],
            dropout=config["dropout"],
        )

    def forward(
        self,
        speech_input: torch.Tensor,
        vision_input: torch.Tensor,
        video_input: torch.Tensor,
        tgt_input: torch.Tensor,
    ) -> torch.Tensor:
        logger.debug("Starting forward pass of MultimodalModel")
        speech_features = self.speech_encoder(speech_input)
        vision_features = self.vision_encoder(vision_input)
        video_features = self.video_encoder(video_input)
        fused_features = self.fusion_module(
            [speech_features, vision_features, video_features]
        )
        output = self.text_decoder(tgt_input, fused_features)
        logger.debug("Completed forward pass of MultimodalModel")
        return output
