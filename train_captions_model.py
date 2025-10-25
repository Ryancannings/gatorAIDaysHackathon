
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

# --- Import the classes from the previous code (copy them above or into another module)
# If you saved my earlier CNN+Attention model as image_captioning.py, then:
from image_captioning import (
    Vocab, CaptionDataset, EncoderCNN, DecoderWithAttention, collate_fn, train_one_epoch
)

# -----------------------------
# 1) Convert captions.txt → JSON format
# -----------------------------
def build_annotations_file(captions_txt="captions.txt", output_json="annotations.json"):
    annotations = []
    with open(captions_txt, "r", encoding="utf-8") as f:
        for line in f:
            if "\t" in line:
                img_id, caption = line.strip().split("\t")
            else:
                parts = line.strip().split(" ", 1)
                if len(parts) < 2:
                    continue
                img_id, caption = parts
            img_name = img_id.split("#")[0]
            caption = caption.strip()
            if len(caption) > 0:
                annotations.append({"image": img_name, "caption": caption})
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(annotations, f)
    print(f"✅ Created {output_json} with {len(annotations)} entries.")
    return output_json

# -----------------------------
# 2) Main training entry point
# -----------------------------
def main():
    # Paths
    IMG_FOLDER = "images"
    CAPTIONS_FILE = "captions.txt"
    JSON_FILE = "annotations.json"

    # Create JSON file from captions.txt if it doesn't exist yet
    if not os.path.exists(JSON_FILE):
        build_annotations_file(CAPTIONS_FILE, JSON_FILE)

    # Load annotations
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    # -----------------------------
    # 3) Vocabulary setup
    # -----------------------------
    all_caps = [a["caption"] for a in annotations]
    vocab = Vocab(freq_threshold=3, max_size=20000)
    vocab.build_vocab(all_caps)
    print(f"✅ Vocabulary built: {len(vocab.itos)} words")

    # -----------------------------
    # 4) Dataset & DataLoader
    # -----------------------------
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225]),
    ])
    dataset = CaptionDataset(IMG_FOLDER, annotations, vocab, transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True,
                            collate_fn=collate_fn, num_workers=4)

    # -----------------------------
    # 5) Model setup
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    encoder = EncoderCNN(encoded_image_size=14).to(device)
    decoder = DecoderWithAttention(
        attention_dim=512,
        embed_dim=512,
        decoder_dim=512,
        vocab_size=len(vocab.itos),
        encoder_dim=2048
    ).to(device)

    # Loss and optimizers
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<pad>"])
    enc_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()), lr=1e-4)
    dec_optimizer = torch.optim.Adam(decoder.parameters(), lr=4e-4)

    # -----------------------------
    # 6) Training loop
    # -----------------------------
    # -----------------------------
    # 6) Training loop
    # -----------------------------
    EPOCHS = 4
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        loss = train_one_epoch(
            encoder, decoder, dataloader,
            criterion, enc_optimizer, dec_optimizer, vocab, device
        )
        print(f"Epoch {epoch + 1} finished. Average Loss: {loss:.4f}")

        # ✅ Save checkpoint at the end of each epoch
        checkpoint = {
            'encoder_state': encoder.state_dict(),
            'decoder_state': decoder.state_dict(),
            'enc_optimizer_state': enc_optimizer.state_dict(),
            'dec_optimizer_state': dec_optimizer.state_dict(),
            'vocab': vocab.__dict__
        }
        torch.save(checkpoint, f"checkpoint_epoch_{epoch + 1}.pth")
        print(f"✅ Saved checkpoint: checkpoint_epoch_{epoch + 1}.pth")


if __name__ == "__main__":
    main()