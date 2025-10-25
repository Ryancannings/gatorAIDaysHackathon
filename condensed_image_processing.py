import os
import json
from collections import Counter
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

# Optional TTS
try:
    import pyttsx3
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False

# ============================================
# 1) Vocabulary Helper
# ============================================
class Vocab:
    def __init__(self, freq_threshold=5, max_size=None):
        self.freq_threshold = freq_threshold
        self.max_size = max_size
        self.itos = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freqs = Counter()

    def tokenize(self, text):
        return text.lower().strip().replace('.', '').replace(',', '').split()

    def build_vocab(self, sentence_list):
        for sent in sentence_list:
            for token in self.tokenize(sent):
                self.freqs[token] += 1
        most_common = [w for w, _ in self.freqs.most_common(self.max_size)]
        idx = len(self.itos)
        for w in most_common:
            if self.freqs[w] < self.freq_threshold:
                continue
            if w not in self.stoi:
                self.stoi[w] = idx
                self.itos[idx] = w
                idx += 1

    def numericalize(self, text):
        tokens = self.tokenize(text)
        return [self.stoi.get(t, self.stoi["<unk>"]) for t in tokens]

# ============================================
# 2) Dataset Loader
# ============================================
class CaptionDataset(Dataset):
    def __init__(self, img_folder, annotations, vocab, transform=None, max_len=30):
        self.img_folder = img_folder
        self.annotations = annotations
        self.vocab = vocab
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])
        ])
        self.max_len = max_len

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_path = os.path.join(self.img_folder, ann['image'])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        caption = ann['caption']
        numericalized = [self.vocab.stoi["<start>"]] + self.vocab.numericalize(caption) + [self.vocab.stoi["<end>"]]
        if len(numericalized) > self.max_len:
            numericalized = numericalized[:self.max_len]
            numericalized[-1] = self.vocab.stoi["<end>"]
        pad_len = self.max_len - len(numericalized)
        numericalized += [self.vocab.stoi["<pad>"]] * pad_len

        return image, torch.tensor(numericalized).long()

# ============================================
# 3) Encoder (ResNet50)
# ============================================
class EncoderCNN(nn.Module):
    def __init__(self, encoded_image_size=14, train_cnn=False):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.fine_tune(train_cnn)

    def forward(self, images):
        features = self.resnet(images)
        features = self.adaptive_pool(features)
        B, C, H, W = features.size()
        return features.permute(0,2,3,1).view(B, H*W, C)

    def fine_tune(self, fine_tune=False):
        for p in self.resnet.parameters():
            p.requires_grad = False
        if fine_tune:
            for c in list(self.resnet.children())[-5:]:
                for p in c.parameters():
                    p.requires_grad = True

# ============================================
# 4) Attention Module
# ============================================
class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.enc_att = nn.Linear(encoder_dim, attention_dim)
        self.dec_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.enc_att(encoder_out)                           # (B, num_pixels, att_dim)
        att2 = self.dec_att(decoder_hidden).unsqueeze(1)           # (B, 1, att_dim)
        att = torch.tanh(att1 + att2)                              # (B, num_pixels, att_dim)
        e = self.full_att(att).squeeze(2)                          # (B, num_pixels)
        alpha = F.softmax(e, dim=1)                                # (B, num_pixels)
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)    # (B, encoder_dim)
        return context, alpha

# ============================================
# 5) Decoder (LSTM + Attention)
# ============================================
class DecoderWithAttention(nn.Module):
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)

    def init_hidden_state(self, encoder_out):
        mean_encoder = encoder_out.mean(dim=1)
        return self.init_h(mean_encoder), self.init_c(mean_encoder)

    def forward(self, encoder_out, captions, lengths):
        batch_size = encoder_out.size(0)
        vocab_size = self.fc.out_features
        embeddings = self.embedding(captions)
        h, c = self.init_hidden_state(encoder_out)
        max_len = captions.size(1)
        preds = torch.zeros(batch_size, max_len, vocab_size).to(encoder_out.device)
        for t in range(max_len):
            emb_t = embeddings[:, t, :]
            context, _ = self.attention(encoder_out, h)
            gate = self.sigmoid(self.f_beta(h))
            context = gate * context
            lstm_input = torch.cat([emb_t, context], dim=1)
            h, c = self.decode_step(lstm_input, (h, c))
            preds[:, t, :] = self.fc(self.dropout(h))
        return preds

# ============================================
# 6) Training Utilities
# ============================================
def collate_fn(batch):
    images, caps = zip(*batch)
    return torch.stack(images), torch.stack(caps), [len(caps[0])] * len(batch)

def train_one_epoch(encoder, decoder, dataloader, criterion, enc_optimizer, dec_optimizer, device, epoch, clip=5.0):
    encoder.train()
    decoder.train()
    total_loss = 0.0
    for i, (imgs, caps, lengths) in enumerate(dataloader):
        imgs, caps = imgs.to(device), caps.to(device)
        enc_out = encoder(imgs)
        scores = decoder(enc_out, caps, lengths)
        B, T, V = scores.size()
        loss = criterion(scores.view(B*T, V), caps.view(B*T))

        if enc_optimizer is not None:
            enc_optimizer.zero_grad()
        dec_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(decoder.parameters(), clip)
        if enc_optimizer is not None:
            nn.utils.clip_grad_norm_(encoder.parameters(), clip)
            enc_optimizer.step()
        dec_optimizer.step()

        total_loss += loss.item()
        if (i + 1) % 50 == 0:
            print(f"Epoch {epoch} | Step {i+1}/{len(dataloader)} | Loss: {loss.item():.4f}")
    return total_loss / len(dataloader)

# ============================================
# 7) Convert captions.txt → JSON
# ============================================
def build_annotations_file(captions_txt="captions.txt", output_json="annotations.json"):
    annotations = []
    with open(captions_txt, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t") if "\t" in line else line.strip().split(" ", 1)
            if len(parts) < 2:
                continue
            img_id, caption = parts
            img_name = img_id.split("#")[0]
            if caption.strip():
                annotations.append({"image": img_name, "caption": caption.strip()})
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(annotations, f)
    print(f"✅ Created {output_json} with {len(annotations)} entries.")
    return output_json

# ============================================
# 8) Greedy Inference + TTS
# ============================================
def generate_caption(encoder, decoder, image_tensor, vocab, device, max_len=30):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        enc_out = encoder(image_tensor.unsqueeze(0).to(device))   # (1, num_pixels, enc_dim)
        h, c = decoder.init_hidden_state(enc_out)
        word = torch.tensor([vocab.stoi["<start>"]]).to(device)
        caption_words = []
        for _ in range(max_len):
            emb = decoder.embedding(word).squeeze(0)               # (embed_dim)
            context, _ = decoder.attention(enc_out, h)             # (1, enc_dim) -> squeezes in Attention
            if context.dim() == 2:
                context = context.squeeze(0)                       # (enc_dim)
            gate = decoder.sigmoid(decoder.f_beta(h))
            context = gate * context
            lstm_input = torch.cat([emb, context], dim=0).unsqueeze(0)  # (1, embed+enc)
            h, c = decoder.decode_step(lstm_input, (h, c))
            out = decoder.fc(h)                                    # (1, vocab)
            pred = out.argmax(dim=1)
            idx = pred.item()
            word = pred
            if idx == vocab.stoi["<end>"]:
                break
            caption_words.append(vocab.itos.get(idx, "<unk>"))
        return " ".join(caption_words)

def speak_text(text):
    if TTS_AVAILABLE:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    else:
        print("[TTS unavailable] Caption:", text)

# ============================================
# 9) Main: train and optionally run inference
# ============================================
def main():
    print("🎨 Art Captioning Model — with optional TTS\n")
    IMG_FOLDER = "images"
    CAPTIONS_FILE = "captions.txt"
    JSON_FILE = "annotations.json"
    CHECKPOINT_DIR = "checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    if not os.path.exists(JSON_FILE):
        build_annotations_file(CAPTIONS_FILE, JSON_FILE)

    with open(JSON_FILE, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    all_caps = [a["caption"] for a in annotations]
    vocab = Vocab(freq_threshold=3, max_size=20000)
    vocab.build_vocab(all_caps)
    print(f"✅ Vocabulary built: {len(vocab.itos)} words")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    dataset = CaptionDataset(IMG_FOLDER, annotations, vocab, transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    encoder = EncoderCNN(encoded_image_size=14).to(device)
    decoder = DecoderWithAttention(512, 512, 512, len(vocab.itos), 2048).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<pad>"])
    enc_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()), lr=1e-4)
    dec_optimizer = torch.optim.Adam(decoder.parameters(), lr=4e-4)

    # TRAIN
    EPOCHS = 3
    for epoch in range(1, EPOCHS + 1):
        print(f"\n🕐 Epoch {epoch}/{EPOCHS}")
        loss = train_one_epoch(encoder, decoder, dataloader, criterion, enc_optimizer, dec_optimizer, device, epoch)
        print(f"✅ Epoch {epoch} complete. Avg Loss: {loss:.4f}")

        ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pth")
        checkpoint = {
            "encoder_state": encoder.state_dict(),
            "decoder_state": decoder.state_dict(),
            "enc_optimizer_state": enc_optimizer.state_dict(),
            "dec_optimizer_state": dec_optimizer.state_dict(),
            "vocab": vocab.__dict__,
        }
        torch.save(checkpoint, ckpt_path)
        print(f"💾 Saved checkpoint: {ckpt_path}")

    # INFERENCE & TTS demo on a sample image
    sample_img = None
    # pick first image in annotations if exists
    if len(annotations) > 0:
        sample_img = os.path.join(IMG_FOLDER, annotations[0]["image"])
    if sample_img and os.path.exists(sample_img):
        print("\n🔎 Running demo inference on:", sample_img)
        img = Image.open(sample_img).convert("RGB")
        img_t = transform(img)
        caption = generate_caption(encoder, decoder, img_t, vocab, device, max_len=30)
        print("Generated caption:", caption)
        speak_text(caption)
    else:
        print("\n[Demo inference skipped] No sample image found in images/ or annotations is empty.")

if __name__ == "__main__":
    main()
