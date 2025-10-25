import os
import json
from collections import Counter
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import resnet50, ResNet50_Weights

# ---------------------------
# 1) Vocabulary helper
# ---------------------------
class Vocab:
    def __init__(self, freq_threshold=5, max_size=None):
        self.freq_threshold = freq_threshold
        self.max_size = max_size
        self.itos = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.stoi = {v:k for k,v in self.itos.items()}
        self.freqs = Counter()

    def tokenize(self, text):
        return text.lower().strip().replace('.', '').replace(',', '').split()

    def build_vocab(self, sentence_list):
        for sent in sentence_list:
            for token in self.tokenize(sent):
                self.freqs[token] += 1
        most_common = [w for w,c in self.freqs.most_common(self.max_size)]
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

# ---------------------------
# 2) Dataset
# ---------------------------
class CaptionDataset(Dataset):
    def __init__(self, img_folder, annotations, vocab, transform=None, max_len=30):
        self.img_folder = img_folder
        self.annotations = annotations
        self.vocab = vocab
        self.transform = transform or transforms.Compose([
            transforms.Resize((256,256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
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

# ---------------------------
# 3) Encoder (ResNet50)
# ---------------------------
class EncoderCNN(nn.Module):
    def __init__(self, encoded_image_size=14, train_cnn=False):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        modules = list(resnet.children())[:-2]  # remove avgpool & fc
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.fine_tune(train_cnn)

    def forward(self, images):
        features = self.resnet(images)
        features = self.adaptive_pool(features)
        B, C, H, W = features.size()
        features = features.permute(0,2,3,1).view(B, H*W, C)
        return features

    def fine_tune(self, fine_tune=False):
        for p in self.resnet.parameters():
            p.requires_grad = False
        if fine_tune:
            for c in list(self.resnet.children())[-5:]:
                for p in c.parameters():
                    p.requires_grad = True

# ---------------------------
# 4) Attention module
# ---------------------------
class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.enc_att = nn.Linear(encoder_dim, attention_dim)
        self.dec_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.enc_att(encoder_out)
        att2 = self.dec_att(decoder_hidden).unsqueeze(1)
        att = torch.tanh(att1 + att2)
        e = self.full_att(att).squeeze(2)
        alpha = F.softmax(e, dim=1)
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return context, alpha

# ---------------------------
# 5) Decoder (LSTM) with attention
# ---------------------------
class DecoderWithAttention(nn.Module):
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)

    def init_hidden_state(self, encoder_out):
        mean_encoder = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder)
        c = self.init_c(mean_encoder)
        return h, c

    def forward(self, encoder_out, captions, lengths):
        batch_size = encoder_out.size(0)
        vocab_size = self.fc.out_features
        embeddings = self.embedding(captions)
        h, c = self.init_hidden_state(encoder_out)
        max_len = captions.size(1)
        preds = torch.zeros(batch_size, max_len, vocab_size).to(encoder_out.device)
        alphas = torch.zeros(batch_size, max_len, encoder_out.size(1)).to(encoder_out.device)

        for t in range(max_len):
            emb_t = embeddings[:, t, :]
            context, alpha = self.attention(encoder_out, h)
            gate = self.sigmoid(self.f_beta(h))
            context = gate * context
            lstm_input = torch.cat([emb_t, context], dim=1)
            h, c = self.decode_step(lstm_input, (h, c))
            output = self.fc(self.dropout(h))
            preds[:, t, :] = output
            alphas[:, t, :] = alpha
        return preds, alphas

# ---------------------------
# 6) Collate & train helpers
# ---------------------------
def collate_fn(batch):
    images, caps = zip(*batch)
    images = torch.stack(images)
    caps = torch.stack(caps)
    lengths = [len(caps[0])] * len(batch)
    return images, caps, lengths

def train_one_epoch(encoder, decoder, dataloader, criterion, enc_optimizer, dec_optimizer, device, epoch, clip=5.0):
    encoder.train()
    decoder.train()
    total_loss = 0.0
    for i, (imgs, caps, lengths) in enumerate(dataloader):
        imgs, caps = imgs.to(device), caps.to(device)
        enc_out = encoder(imgs)
        scores, alphas = decoder(enc_out, caps, lengths)
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
        if (i+1) % 100 == 0:
            print(f"Epoch {epoch} Iter {i+1}/{len(dataloader)} Loss: {loss.item():.4f}")
    return total_loss / len(dataloader)

# ---------------------------
# 7) Greedy inference
# ---------------------------
def generate_caption(encoder, decoder, image_tensor, vocab, device, max_len=30):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        enc_out = encoder(image_tensor.unsqueeze(0).to(device))
        h, c = decoder.init_hidden_state(enc_out)
        word = torch.tensor([vocab.stoi["<start>"]]).to(device)
        caption = []
        for t in range(max_len):
            emb = decoder.embedding(word).squeeze(0)
            context, alpha = decoder.attention(enc_out, h)
            gate = decoder.sigmoid(decoder.f_beta(h))
            context = gate * context
            lstm_input = torch.cat([emb, context], dim=0).unsqueeze(0)
            h, c = decoder.decode_step(lstm_input, (h,c))
            out = decoder.fc(h)
            pred = out.argmax(dim=1)
            word = pred
            idx = pred.item()
            if idx == vocab.stoi["<end>"]:
                break
            caption.append(vocab.itos.get(idx, "<unk>"))
        return " ".join(caption)

# ---------------------------
# 8) Main training / example
# ---------------------------
def main():
    IMG_FOLDER = "./images"  # change to your folder
    ANNS_JSON = "./annotations.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(ANNS_JSON, "r") as f:
        annotations = json.load(f)

    all_caps = [a['caption'] for a in annotations]
    vocab = Vocab(freq_threshold=3, max_size=20000)
    vocab.build_vocab(all_caps)
    print(f"Vocab size: {len(vocab.itos)}")

    dataset = CaptionDataset(IMG_FOLDER, annotations, vocab)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=2)

    # Models
    train_cnn = False  # set True to fine-tune ResNet
    encoder = EncoderCNN(encoded_image_size=14, train_cnn=train_cnn).to(device)
    decoder = DecoderWithAttention(attention_dim=512, embed_dim=512, decoder_dim=512,
                                   vocab_size=len(vocab.itos), encoder_dim=2048).to(device)

    enc_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()), lr=1e-4) if train_cnn else None
    dec_optimizer = torch.optim.Adam(decoder.parameters(), lr=4e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<pad>"])

    epochs = 10
    for epoch in range(1, epochs+1):
        loss = train_one_epoch(encoder, decoder, dataloader, criterion, enc_optimizer, dec_optimizer, device, epoch)
        print(f"Epoch {epoch} avg loss: {loss:.4f}")
        # Save checkpoint
        torch