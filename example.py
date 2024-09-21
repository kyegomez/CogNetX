import torch
from cognetx.model import CogNetX

if __name__ == "__main__":
    # Example configuration and usage
    config = {
        "speech_input_dim": 80,  # For example, 80 Mel-filterbank features
        "speech_num_layers": 4,
        "speech_num_heads": 8,
        "encoder_dim": 256,
        "decoder_dim": 512,
        "vocab_size": 10000,
        "embedding_dim": 512,
        "decoder_num_layers": 6,
        "decoder_num_heads": 8,
        "dropout": 0.1,
        "depthwise_conv_kernel_size": 31,
    }

    model = CogNetX(config)

    # Dummy inputs
    batch_size = 2
    speech_input = torch.randn(
        batch_size, 500, config["speech_input_dim"]
    )  # (batch_size, time_steps, feature_dim)
    vision_input = torch.randn(
        batch_size, 3, 224, 224
    )  # (batch_size, 3, H, W)
    video_input = torch.randn(
        batch_size, 3, 16, 112, 112
    )  # (batch_size, 3, time_steps, H, W)
    tgt_input = torch.randint(
        0, config["vocab_size"], (20, batch_size)
    )  # (tgt_seq_len, batch_size)

    # Forward pass
    output = model(speech_input, vision_input, video_input, tgt_input)
    print(
        output.shape
    )  # Expected: (tgt_seq_len, batch_size, vocab_size)
