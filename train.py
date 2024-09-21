import os
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from datasets import load_dataset
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoTokenizer

from cognetx import CogNetX

# Assume the MultimodalModel and its components are defined as before
# Include the code for MultimodalModel from previous steps or import it if defined in another module

# Configure loguru logger
logger.add(
    "training.log", format="{time} {level} {message}", level="INFO"
)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters and configuration
config = {
    "speech_input_dim": 80,
    "speech_num_layers": 4,
    "speech_num_heads": 8,
    "encoder_dim": 256,
    "decoder_dim": 512,
    "vocab_size": 30522,  # Using BERT tokenizer vocab size
    "embedding_dim": 512,
    "decoder_num_layers": 6,
    "decoder_num_heads": 8,
    "dropout": 0.1,
    "depthwise_conv_kernel_size": 31,
    "batch_size": 8,
    "num_epochs": 5,
    "learning_rate": 1e-4,
    "save_path": "./model_checkpoints",
}

# Ensure the save path exists
os.makedirs(config["save_path"], exist_ok=True)

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


# Define custom dataset
class MultimodalDataset(Dataset):
    def __init__(self):
        # Load datasets
        self.speech_dataset = load_dataset(
            "librispeech_asr", "clean", split="train.100"
        )
        self.image_dataset = load_dataset(
            "ms_coco", "2014", split="train"
        )
        self.video_dataset = load_dataset("msr_vtt", split="train")

        # Preprocessing transforms for images and videos
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )
        self.video_transform = transforms.Compose(
            [
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
            ]
        )

        # Truncate datasets to the smallest size for alignment
        min_size = min(
            len(self.speech_dataset),
            len(self.image_dataset),
            len(self.video_dataset),
        )
        self.size = min_size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Load speech data
        speech_sample = self.speech_dataset[idx]
        speech_audio = speech_sample["audio"]["array"]
        speech_text = speech_sample["text"]

        # Load image data
        image_sample = self.image_dataset[idx]
        image = image_sample["image"]
        image_caption = image_sample["caption"]

        # Load video data
        video_sample = self.video_dataset[idx]
        video = video_sample["video"]
        video_caption = video_sample["text"]

        # Align text output (for demonstration, use speech transcription)
        text_output = speech_text

        # Preprocess speech
        speech_input = self.process_speech(speech_audio)

        # Preprocess image
        image_input = self.process_image(image)

        # Preprocess video
        video_input = self.process_video(video)

        # Tokenize text_output
        tgt_input = self.process_text(text_output)

        return {
            "speech_input": speech_input,
            "vision_input": image_input,
            "video_input": video_input,
            "tgt_input": tgt_input,
        }

    def process_speech(self, audio: Any) -> torch.Tensor:
        # Convert audio to Mel-spectrogram
        speech_tensor = torch.tensor(audio, dtype=torch.float32)
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_mels=config["speech_input_dim"],
        )(speech_tensor)
        # Transpose to (time_steps, feature_dim)
        speech_input = mel_spectrogram.transpose(0, 1)
        return speech_input

    def process_image(self, image: Any) -> torch.Tensor:
        image = image.convert("RGB")
        image_input = self.image_transform(image)
        return image_input

    def process_video(self, video: Any) -> torch.Tensor:
        # For demonstration, use the first frame as a placeholder
        video_frames = []
        for frame in video:
            frame = frame.convert("RGB")
            frame_tensor = self.video_transform(frame)
            video_frames.append(frame_tensor)
        video_input = torch.stack(video_frames, dim=1)  # (C, T, H, W)
        return video_input

    def process_text(self, text: str) -> torch.Tensor:
        # Tokenize text
        tokens = tokenizer.encode(text, add_special_tokens=True)
        tgt_input = torch.tensor(tokens, dtype=torch.long)
        return tgt_input


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Collate function to handle variable-length sequences
    speech_inputs = [item["speech_input"] for item in batch]
    vision_inputs = [item["vision_input"] for item in batch]
    video_inputs = [item["video_input"] for item in batch]
    tgt_inputs = [item["tgt_input"] for item in batch]

    # Pad sequences
    speech_inputs_padded = nn.utils.rnn.pad_sequence(
        speech_inputs, batch_first=True
    )
    tgt_inputs_padded = nn.utils.rnn.pad_sequence(
        tgt_inputs, batch_first=False
    )

    vision_inputs_tensor = torch.stack(vision_inputs)
    video_inputs_tensor = torch.stack(video_inputs)

    return {
        "speech_input": speech_inputs_padded.to(device),
        "vision_input": vision_inputs_tensor.to(device),
        "video_input": video_inputs_tensor.to(device),
        "tgt_input": tgt_inputs_padded.to(device),
    }


# Initialize dataset and dataloader
dataset = MultimodalDataset()
dataloader = DataLoader(
    dataset,
    batch_size=config["batch_size"],
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4,
)

# Initialize model, optimizer, and loss function
model = CogNetX(config).to(device)
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# Training loop
for epoch in range(config["num_epochs"]):
    logger.info(f"Starting epoch {epoch + 1}/{config['num_epochs']}")
    model.train()
    total_loss = 0.0
    for batch_idx, batch in enumerate(dataloader):
        speech_input = batch["speech_input"]
        vision_input = batch["vision_input"]
        video_input = batch["video_input"]
        tgt_input = batch["tgt_input"]

        # Shift tgt_input for teacher forcing
        tgt_input_in = tgt_input[:-1, :]
        tgt_input_out = tgt_input[1:, :]

        # Forward pass
        optimizer.zero_grad()
        output = model(
            speech_input, vision_input, video_input, tgt_input_in
        )

        # Flatten output and target tensors
        output_flat = output.view(-1, config["vocab_size"])
        tgt_input_out_flat = tgt_input_out.reshape(-1)

        # Compute loss
        loss = criterion(output_flat, tgt_input_out_flat)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (batch_idx + 1) % 10 == 0:
            avg_loss = total_loss / 10
            logger.info(
                f"Epoch [{epoch + 1}/{config['num_epochs']}], "
                f"Step [{batch_idx + 1}/{len(dataloader)}], "
                f"Loss: {avg_loss:.4f}"
            )
            total_loss = 0.0

    # Save model checkpoint
    checkpoint_path = os.path.join(
        config["save_path"], f"model_epoch_{epoch + 1}.pt"
    )
    torch.save(model.state_dict(), checkpoint_path)
    logger.info(f"Saved model checkpoint to {checkpoint_path}")
