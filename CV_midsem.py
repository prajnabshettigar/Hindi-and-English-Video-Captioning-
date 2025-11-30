#!/usr/bin/env python
# coding: utf-8

# In[42]:


import os
import json
import random
import math
import glob
import numpy as np
from tqdm import tqdm
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
import torchvision.models as models
import cv2

import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
nltk.download('punkt')


# In[43]:


VIDEO_DIR = os.path.join("dataset/data/MSRVTT/MSRVTT/videos")  
CAPTIONS_FILE = os.path.join("dataset/data/MSRVTT/MSRVTT/annotation/MSR_VTT.json") 

# Hyperparams (tweak)
SAMPLE_FRAMES = 16       
FEATURE_DIM = 2048       
ENC_HIDDEN = 512
DEC_HIDDEN = 512
EMBED_SIZE = 512
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)


# In[ ]:


import numpy as np

file_path = "/kaggle/input/msr-vtt-global-motion-350/global_features/video4.npy"   
data = np.load(file_path, allow_pickle=True)

print("Type:", type(data))
print("Shape:", data.shape)
print("Data type:", data.dtype)

print("\nSample values:\n", data[:4])   # print first 2 items or frames


# In[5]:


import json
import pandas as pd

# Path to your annotation file
anno_file = "/kaggle/input/msr-vtt/MSR_VTT.json"

# Load JSON
with open(anno_file, "r") as f:
    data = json.load(f)

# Convert annotations into DataFrame
df = pd.DataFrame(data["annotations"])

# Show first 5 rows
print(df.head())


# In[44]:


#  vocabulary from captions
from nltk.tokenize import word_tokenize

CAPTIONS_FILE = os.path.join("/kaggle/input/msr-vtt/captions.json") 

class Vocab:
    def __init__(self, freq_threshold=1, max_size=None):
        self.freq_threshold = freq_threshold
        self.max_size = max_size
        self.word2idx = {}
        self.idx2word = {}
        self.pad_token = "<PAD>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        self.unk_token = "<UNK>"
        for i,w in enumerate([self.pad_token, self.bos_token, self.eos_token, self.unk_token]):
            self.word2idx[w] = i
        self.idx2word = {i:w for w,i in self.word2idx.items()}

    def build_vocab(self, captions_dict):
        counter = Counter()
        for vid, caps in captions_dict.items():
            for c in caps:
                tokens = [t.lower() for t in word_tokenize(c)]
                counter.update(tokens)
        # filter
        words = [w for w,c in counter.items() if c >= self.freq_threshold]
        words = sorted(words, key=lambda w: (-counter[w], w))
        if self.max_size:
            words = words[:self.max_size - len(self.word2idx)]
        idx = len(self.word2idx)
        for w in words:
            self.word2idx[w] = idx
            self.idx2word[idx] = w
            idx += 1

    def numericalize(self, text):
        tokens = [t.lower() for t in word_tokenize(text)]
        nums = [self.word2idx.get(t, self.word2idx[self.unk_token]) for t in tokens]
        return [self.word2idx[self.bos_token]] + nums + [self.word2idx[self.eos_token]]

# Load captions.json
with open(CAPTIONS_FILE, 'r', encoding='utf-8') as f:
    captions = json.load(f)

vocab = Vocab(freq_threshold=2, max_size=20000)
vocab.build_vocab(captions)
print("Vocab size:", len(vocab.word2idx))


# In[45]:


# Dataset class
FEATURES_DIR = "/kaggle/input/msr-vtt/features/features"
class MSRVTTDataset(Dataset):
    def __init__(self, captions_dict, features_dir, vocab, sample_frames=SAMPLE_FRAMES, max_caption_len=30):
        self.items = []  # (video_filename, caption_text)
        for vid, caps in captions_dict.items():
            for c in caps:
                self.items.append((vid, c))
        self.features_dir = features_dir
        self.vocab = vocab
        self.sample_frames = sample_frames
        self.max_caption_len = max_caption_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        vid, cap = self.items[idx]
        feat_path = os.path.join(self.features_dir, vid.replace('.mp4', '.npy'))
        feats = np.load(feat_path)  # (T, D)
        # pad/truncate frames (should already be SAMPLE_FRAMES)
        if feats.shape[0] < self.sample_frames:
            pad = np.zeros((self.sample_frames - feats.shape[0], feats.shape[1]), dtype=np.float32)
            feats = np.concatenate([feats, pad], axis=0)
        else:
            feats = feats[:self.sample_frames]
        numer = self.vocab.numericalize(cap)
        if len(numer) > self.max_caption_len:
            numer = numer[:self.max_caption_len-1] + [self.vocab.word2idx[self.vocab.eos_token]]
        cap_len = len(numer)
        # pad caption
        pad_len = self.max_caption_len - cap_len
        if pad_len > 0:
            numer = numer + [self.vocab.word2idx[self.vocab.pad_token]] * pad_len
        return torch.FloatTensor(feats), torch.LongTensor(numer), cap_len

def collate_fn(batch):
    feats = torch.stack([b[0] for b in batch], dim=0)  # (B, T, D)
    caps = torch.stack([b[1] for b in batch], dim=0)
    cap_lens = torch.LongTensor([b[2] for b in batch])
    return feats, caps, cap_lens

# split dataset (simple random split)
items = list(captions.items())
random.seed(42)
random.shuffle(items)
n = len(items)
train_items = dict(items[:int(0.8*n)])
val_items = dict(items[int(0.8*n):int(0.9*n)])
test_items = dict(items[int(0.9*n):])

train_ds = MSRVTTDataset(train_items, FEATURES_DIR, vocab)
val_ds   = MSRVTTDataset(val_items, FEATURES_DIR, vocab)
test_ds  = MSRVTTDataset(test_items, FEATURES_DIR, vocab)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4)

print("Train size:", len(train_ds), "Val size:", len(val_ds), "Test size:", len(test_ds))


# In[ ]:


#  Model-1)BiLSTM Encoder+LSTM Decoder

class EncoderRNN(nn.Module):
    def __init__(self, feat_size, hidden_size, num_layers=1, bidirectional=True,dropout=0.1):
        super().__init__()
        self.rnn = nn.LSTM(input_size=feat_size, hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.output_size = hidden_size * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, feats):
        # feats: (B, T, D)
        outputs, (h_n, c_n) = self.rnn(feats)  # outputs: (B, T, hidden*dir)
        return outputs, (h_n, c_n)

class LuongAttention(nn.Module):
    def __init__(self, enc_dim, dec_dim):
        super().__init__()
        self.attn = nn.Linear(enc_dim, dec_dim)

    def forward(self, decoder_hidden, encoder_outputs, mask=None):

        proj = self.attn(encoder_outputs)  # (B, T, dec_dim)
        # up: (B, T)
        scores = torch.bmm(proj, decoder_hidden.unsqueeze(2)).squeeze(2)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(scores, dim=1)  # (B, T)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (B, enc_dim)
        return context, attn_weights



class DecoderWithAttention(nn.Module):
    def __init__(self, embed_size, enc_dim, dec_hidden, vocab_size, num_layers=1,num_heads=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = LuongAttention(enc_dim, dec_hidden)

        self.lstm = nn.LSTMCell(embed_size + enc_dim, dec_hidden)
        self.fc_out = nn.Linear(dec_hidden, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward_step(self, prev_word, last_hidden, last_cell, encoder_outputs):
        # prev_word: (B,) token ids
        emb = self.embedding(prev_word)  # (B, E)

        context, attn_weights = self.attention(last_hidden, encoder_outputs)  # (B, enc_dim), (B, T)

        lstm_input = torch.cat([emb, context], dim=1)
        h, c = self.lstm(lstm_input, (last_hidden, last_cell))
        output = self.fc_out(self.dropout(h))
        return output, h, c, attn_weights

    def forward(self, encoder_outputs, captions, teacher_forcing_ratio=0.9):
        # encoder_outputs: (B, T, enc_dim)
        batch_size = encoder_outputs.size(0)
        max_len = captions.size(1)
        vocab_size = self.fc_out.out_features


        hidden = torch.zeros(batch_size, self.lstm.hidden_size, device=encoder_outputs.device)
        cell = torch.zeros(batch_size, self.lstm.hidden_size, device=encoder_outputs.device)

        outputs = torch.zeros(batch_size, max_len, vocab_size, device=encoder_outputs.device)
        attn_weights_all = []


        input_word = captions[:,0]  # (B,)
        for t in range(1, max_len):
            out, hidden, cell, attn_weights = self.forward_step(input_word, hidden, cell, encoder_outputs)
            outputs[:, t, :] = out
            attn_weights_all.append(attn_weights)
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = out.argmax(1)
            input_word = captions[:, t] if teacher_force else top1
        # outputs: (B, max_len, vocab)
        return outputs, attn_weights_all


# In[ ]:


#Decoder model using BiLSTM

class DecoderWithBiLSTM(nn.Module):
    def __init__(self, embed_size, enc_dim, dec_hidden, vocab_size, num_layers=1, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = LuongAttention(enc_dim, dec_hidden * 2)  # since decoder is bidirectional

        self.bilstm = nn.LSTM(input_size=embed_size + enc_dim,
                              hidden_size=dec_hidden,
                              num_layers=num_layers,
                              batch_first=True,
                              bidirectional=True)

        self.fc_out = nn.Linear(dec_hidden * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_outputs, captions, teacher_forcing_ratio=0.9):
        # encoder_outputs: (B, T, enc_dim)
        batch_size = encoder_outputs.size(0)
        max_len = captions.size(1)
        vocab_size = self.fc_out.out_features

        # Initialize hidden and cell states for both directions
        h0 = torch.zeros(2, batch_size, self.bilstm.hidden_size, device=encoder_outputs.device)
        c0 = torch.zeros(2, batch_size, self.bilstm.hidden_size, device=encoder_outputs.device)

        # Embedding all captions
        embeddings = self.embedding(captions)  # (B, L, E)
        outputs = torch.zeros(batch_size, max_len, vocab_size, device=encoder_outputs.device)
        attn_weights_all = []

        # Loop through each timestep (teacher forcing)
        input_word = captions[:, 0]  # <BOS> token
        hidden, cell = h0, c0

        for t in range(1, max_len):
            emb = self.embedding(input_word).unsqueeze(1)  # (B,1,E)


            # Combine both directions' last layer hidden states
            last_hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)  # (B, dec_hidden*2)
            context, attn_weights = self.attention(last_hidden_cat, encoder_outputs)
            attn_weights_all.append(attn_weights)

            # Concatenate embedding and context
            lstm_input = torch.cat([emb, context.unsqueeze(1)], dim=2)  # (B,1,E+enc_dim)

            # Run one BiLSTM step (we process one token at a time)
            output, (hidden, cell) = self.bilstm(lstm_input, (hidden, cell))  # output: (B,1,2*hidden)

            # Generate next word prediction
            out_vocab = self.fc_out(self.dropout(output.squeeze(1)))  # (B, vocab)
            outputs[:, t, :] = out_vocab

            # Decide whether to use teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = out_vocab.argmax(1)
            input_word = captions[:, t] if teacher_force else top1

        return outputs, attn_weights_all


# Initilaization of Encoder and Decoder

# In[ ]:


enc = EncoderRNN(feat_size=FEATURE_DIM, hidden_size=ENC_HIDDEN, bidirectional=True).to(DEVICE)
# dec = DecoderWithAttention(embed_size=EMBED_SIZE, enc_dim=enc.output_size, dec_hidden=DEC_HIDDEN, vocab_size=len(vocab.word2idx)).to(DEVICE)
dec = DecoderWithBiLSTM(embed_size=EMBED_SIZE, enc_dim=enc.output_size, dec_hidden=DEC_HIDDEN, vocab_size=len(vocab.word2idx)).to(DEVICE)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1,ignore_index=vocab.word2idx[vocab.pad_token])
params = list(enc.parameters()) + list(dec.parameters())
optimizer = optim.Adam(params, lr=LR)

def train_one_epoch(train_loader, enc, dec, optimizer, criterion, device, clip=5.0):
    enc.train(); dec.train()
    running_loss = 0.0
    for feats, caps, cap_lens in tqdm(train_loader):
        feats = feats.to(device)            # (B, T, D)
        caps = caps.to(device)              # (B, L)
        optimizer.zero_grad()
        encoder_outs, _ = enc(feats)       # (B, T, enc_dim)
        outputs, _ = dec(encoder_outs, caps, teacher_forcing_ratio=0.75)  # (B, L, V)
        # shift outputs and targets: ignore the first token (<BOS>)
        outputs = outputs[:,1:,:].contiguous()
        targets = caps[:,1:].contiguous()
        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, clip)
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def evaluate(loader, enc, dec, device):
    enc.eval(); dec.eval()
    all_refs = []
    all_preds = []
    with torch.no_grad():
        for feats, caps, cap_lens in tqdm(loader):
            feats = feats.to(device)
            caps = caps.to(device)
            encoder_outs, _ = enc(feats)
            # greedy decode
            batch_size = feats.size(0)
            hidden = torch.zeros(batch_size, dec.lstm.hidden_size, device=device)
            cell   = torch.zeros(batch_size, dec.lstm.hidden_size, device=device)
            input_word = torch.LongTensor([vocab.word2idx[vocab.bos_token]] * batch_size).to(device)
            preds = [[] for _ in range(batch_size)]
            for t in range(1, caps.size(1)):
                out, hidden, cell, attn = dec.forward_step(input_word, hidden, cell, encoder_outs)
                top1 = out.argmax(1)  # (B,)
                input_word = top1
                for i in range(batch_size):
                    preds[i].append(decoder_token := top1[i].item())
            # convert preds and refs to token lists
            for i in range(batch_size):
                # predicted sentence until EOS or max len
                p = []
                for tok in preds[i]:
                    if tok == vocab.word2idx[vocab.eos_token]:
                        break
                    if tok == vocab.word2idx[vocab.pad_token] or tok == vocab.word2idx[vocab.bos_token]:
                        continue
                    p.append(vocab.idx2word.get(tok, vocab.unk_token))
                all_preds.append(p)

                ref_tokens = []
                target_seq = caps[i].cpu().numpy()
                for tok in target_seq:
                    if tok == vocab.word2idx[vocab.eos_token]:
                        break
                    if tok in (vocab.word2idx[vocab.bos_token], vocab.word2idx[vocab.pad_token]):
                        continue
                    ref_tokens.append(vocab.idx2word.get(int(tok), vocab.unk_token))
                all_refs.append([ref_tokens])
    # BLEU
    smoothie = SmoothingFunction().method4
    bleu4 = corpus_bleu(all_refs, all_preds, weights=(0.25,0.25,0.25,0.25), smoothing_function=smoothie)
    return bleu4, all_refs, all_preds


# In[ ]:


import torch
from tqdm import tqdm
import random
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge import Rouge
from pycocoevalcap.cider.cider import Cider

def evaluate_with_metrics(loader, enc, dec, vocab, device, DEC_HIDDEN=512):
    enc.eval()
    dec.eval()
    all_refs, all_preds = [], []

    rouge = Rouge()
    cider_scorer = Cider()

    with torch.no_grad():
        for feats, caps, cap_lens in tqdm(loader, desc="Evaluating"):
            feats, caps = feats.to(device), caps.to(device)
            B = feats.size(0)

            # Encoder forward
            encoder_outs, _ = enc(feats)

            # Initialize hidden and cell
            hidden = torch.zeros(B, DEC_HIDDEN, device=device)
            cell = torch.zeros(B, DEC_HIDDEN, device=device)

            # Start with <BOS> token
            input_word = torch.LongTensor([vocab.word2idx[vocab.bos_token]] * B).to(device)

            preds = [[] for _ in range(B)]

            # Greedy decoding
            max_len = caps.size(1)
            for t in range(1, max_len):
                out, hidden, cell, attn_weights = dec.forward_step(
                    input_word, hidden, cell, encoder_outs
                )
                top1 = out.argmax(1)
                input_word = top1
                for i in range(B):
                    preds[i].append(top1[i].item())

            # Convert predicted tokens → words
            for i in range(B):
                pred_tokens = []
                for tok in preds[i]:
                    if tok in (vocab.word2idx[vocab.pad_token], vocab.word2idx[vocab.bos_token]):
                        continue
                    if tok == vocab.word2idx[vocab.eos_token]:
                        break
                    pred_tokens.append(vocab.idx2word.get(tok, vocab.unk_token))
                all_preds.append(pred_tokens)

                # Reference captions
                ref_tokens = []
                for tok in caps[i].cpu().numpy():
                    if tok in (vocab.word2idx[vocab.pad_token], vocab.word2idx[vocab.bos_token]):
                        continue
                    if tok == vocab.word2idx[vocab.eos_token]:
                        break
                    ref_tokens.append(vocab.idx2word.get(int(tok), vocab.unk_token))
                all_refs.append([ref_tokens])


    # Compute Metrics

    smoothie = SmoothingFunction().method4
    bleu4 = corpus_bleu(all_refs, all_preds, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

    refs_str = [' '.join(ref[0]) for ref in all_refs]
    preds_str = [' '.join(pred) for pred in all_preds]

    rouge_scores = rouge.get_scores(preds_str, refs_str, avg=True)
    cider_score, _ = cider_scorer.compute_score(
        {i: [refs_str[i]] for i in range(len(refs_str))},
        {i: [preds_str[i]] for i in range(len(preds_str))}
    )

    print(f"\nBLEU-4: {bleu4:.4f}")
    print(f"ROUGE-L: {rouge_scores['rouge-l']['f']:.4f}")
    print(f"CIDEr: {cider_score:.4f}")

    return bleu4, rouge_scores, cider_score, all_refs, all_preds


# Training and testing 1st Model BiLSTM Enc & LSTM Dec +Luong +label smoothing+dropout+all 3 metrics

# In[ ]:


#Training Loop

best_bleu = 0.0

for epoch in range(1, EPOCHS + 1):
    #Training 
    train_loss = train_one_epoch(train_loader, enc, dec, optimizer, criterion, DEVICE)
    print(f"Epoch {epoch} | Train Loss: {train_loss:.4f}")

    # Validation 
    bleu, rouge_scores, cider, _, _ = evaluate_with_metrics(val_loader, enc, dec, vocab, DEVICE, DEC_HIDDEN)
    rouge_l_f = rouge_scores['rouge-l']['f']  

    print(f"[Epoch {epoch}] Validation Metrics:")
    print(f"  BLEU-4 = {bleu:.4f}")
    print(f"  ROUGE-L (F1) = {rouge_l_f:.4f}")
    print(f"  CIDEr = {cider:.4f}")


    if bleu > best_bleu:
        best_bleu = bleu
        torch.save({
            'enc_state': enc.state_dict(),
            'dec_state': dec.state_dict(),
            'vocab': vocab.word2idx
        }, "best_checkpoint.pth")
        print("✅ Saved new best checkpoint.")

print(f"\nTraining complete. Best BLEU-4 achieved: {best_bleu:.4f}")


# In[ ]:


import torch
import numpy as np
import os
import random
from tqdm import tqdm

# Load the best saved checkpoint

ckpt = torch.load("best_checkpoint.pth", map_location=DEVICE)
enc.load_state_dict(ckpt['enc_state'])
dec.load_state_dict(ckpt['dec_state'])
print(" Loaded checkpoint successfully.")



# Evaluate on test set

print("Running test evaluation (BLEU, ROUGE, CIDEr)...")
test_bleu, test_rouge, test_cider ,_,_= evaluate_with_metrics(test_loader, enc, dec, vocab, DEVICE,DEC_HIDDEN)
# Extract the ROUGE-L F1 score
test_rouge_l = test_rouge['rouge-l']['f']

print(f"\n📊 Test Metrics:")
print(f"BLEU-4  = {test_bleu:.4f}")
print(f"ROUGE-L = {test_rouge_l:.4f}")
print(f"CIDEr   = {test_cider:.4f}")



def generate_caption_for_video(video_feat, enc, dec, vocab, device, max_len=20):
    enc.eval()
    dec.eval()

    feat_tensor = torch.tensor(video_feat).unsqueeze(0).float().to(device)  # (1, T, D)
    with torch.no_grad():

        encoder_outs, _ = enc(feat_tensor)

        hidden = torch.zeros(1, dec.lstm.hidden_size, device=device)
        cell = torch.zeros(1, dec.lstm.hidden_size, device=device)

        input_word = torch.LongTensor([vocab.word2idx[vocab.bos_token]]).to(device)
        generated_tokens = []

        for _ in range(max_len):
            out, hidden, cell, attn_weights = dec.forward_step(
                input_word, hidden, cell, encoder_outs
            )
            next_word = out.argmax(1).item()

            if next_word == vocab.word2idx[vocab.eos_token]:
                break

            generated_tokens.append(vocab.idx2word.get(next_word, vocab.unk_token))
            input_word = torch.LongTensor([next_word]).to(device)

    return " ".join(generated_tokens)


# few random test samples

sample_videos = random.sample(list(test_items.keys()), 4)

print("\n --- Random Sample Predictions ---")
for i, vid in enumerate(sample_videos):
    sample_feat_path = os.path.join(FEATURES_DIR, vid.replace('.mp4', '.npy'))
    sample_feat = np.load(sample_feat_path)

    generated_caption = generate_caption_for_video(sample_feat, enc, dec, vocab, DEVICE)
    references = test_items[vid]  

    print(f"\n Video {i+1}: {vid}")
    print(f"Generated Caption: {generated_caption}")
    print("Reference Captions:")
    for j, ref in enumerate(references[:3]):  
        print(f"  Ref {j+1}: {ref}")


# In[ ]:


#BiLSTM Encoder(+dropout) decoder+ label smoothening
#  Training loop
EPOCHS=15
best_bleu = 0.0
for epoch in range(1, EPOCHS+1):
    train_loss = train_one_epoch(train_loader, enc, dec, optimizer, criterion, DEVICE)
    print(f"Epoch {epoch} train loss: {train_loss:.4f}")
    bleu, _, _ = evaluate(val_loader, enc, dec, DEVICE)
    print(f"Validation BLEU-4: {bleu:.4f}")
    if bleu > best_bleu:
        best_bleu = bleu
        torch.save({
            'enc_state': enc.state_dict(),
            'dec_state': dec.state_dict(),
            'vocab': vocab.word2idx
        }, "best_checkpoint.pth")
        print("Saved best checkpoint.")


# In[18]:


#  Greedy inference for a single video, and test BLEU
def generate_caption_for_video(feat_np, enc, dec, vocab, device, max_len=30):
    enc.eval(); dec.eval()
    with torch.no_grad():
        feats = torch.FloatTensor(feat_np).unsqueeze(0).to(device)  # (1, T, D)
        encoder_outs, _ = enc(feats)
        hidden = torch.zeros(1, dec.lstm.hidden_size, device=device)
        cell   = torch.zeros(1, dec.lstm.hidden_size, device=device)
        input_word = torch.LongTensor([vocab.word2idx[vocab.bos_token]]).to(device)
        pred_tokens = []
        for t in range(max_len):
            out, hidden, cell, attn = dec.forward_step(input_word, hidden, cell, encoder_outs)
            top1 = out.argmax(1).item()
            if top1 == vocab.word2idx[vocab.eos_token]:
                break
            if top1 not in (vocab.word2idx[vocab.pad_token], vocab.word2idx[vocab.bos_token]):
                pred_tokens.append(vocab.idx2word.get(top1, vocab.unk_token))
            input_word = torch.LongTensor([top1]).to(device)
    return " ".join(pred_tokens)

# Load checkpoint and test
ckpt = torch.load("best_checkpoint.pth", map_location=DEVICE)
enc.load_state_dict(ckpt['enc_state'])
dec.load_state_dict(ckpt['dec_state'])
print("Loaded checkpoint. Running test evaluation (BLEU-4)...")
test_bleu, refs, preds = evaluate(test_loader, enc, dec, DEVICE)
print("Test BLEU-4:", test_bleu)


sample_feat_path = os.path.join(FEATURES_DIR, list(test_items.keys())[0].replace('.mp4', '.npy'))
sample_feat = np.load(sample_feat_path)
print("Generated:", generate_caption_for_video(sample_feat, enc, dec, vocab, DEVICE))
print("Reference(s):", test_items[list(test_items.keys())[0]][:3])  # first few refs


# In[ ]:


# lstm(+dropout)
ckpt = torch.load("best_checkpoint.pth", map_location=DEVICE)
enc.load_state_dict(ckpt['enc_state'])
dec.load_state_dict(ckpt['dec_state'])
print("Loaded checkpoint. Running test evaluation (BLEU-4)...")
test_bleu, refs, preds = evaluate(test_loader, enc, dec, DEVICE)
print("Test BLEU-4:", test_bleu)

# Show results for 3–4 videos
print("\n--- Sample Predictions ---")
for i, vid in enumerate(list(test_items.keys())[:4]):  # first 4 test videos
    sample_feat_path = os.path.join(FEATURES_DIR, vid.replace('.mp4', '.npy'))
    sample_feat = np.load(sample_feat_path)

    # Generate caption
    generated = generate_caption_for_video(sample_feat, enc, dec, vocab, DEVICE)

    # Print result
    print(f"\nVideo {i+1}: {vid}")
    print("Generated:", generated)
    print("Reference(s):", test_items[vid][:3])  # show first 3 references


# In[23]:


import random

# Load checkpoint and test
ckpt = torch.load("best_checkpoint.pth", map_location=DEVICE)
enc.load_state_dict(ckpt['enc_state'])
dec.load_state_dict(ckpt['dec_state'])
print("Loaded checkpoint. Running test evaluation (BLEU-4)...")
test_bleu, refs, preds = evaluate(test_loader, enc, dec, DEVICE)
print("Test BLEU-4:", test_bleu)

# Pick 4 random test videos
sample_videos = random.sample(list(test_items.keys()), 4)

print("\n--- Random Sample Predictions ---")
for i, vid in enumerate(sample_videos):
    sample_feat_path = os.path.join(FEATURES_DIR, vid.replace('.mp4', '.npy'))
    sample_feat = np.load(sample_feat_path)

    # Generate caption
    generated = generate_caption_for_video(sample_feat, enc, dec, vocab, DEVICE)

    # References (original captions)
    references = test_items[vid]

    print(f"\nVideo {i+1}: {vid}")
    print("Generated Caption:", generated)
    print("Original Captions:")
    for j, ref in enumerate(references[:3]):  # show first 3 refs
        print(f"  Ref {j+1}: {ref}")


# In[ ]:


#BiLSTM
import torch
from tqdm import tqdm
import random
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge import Rouge
from pycocoevalcap.cider.cider import Cider

def evaluate_with_metrics(loader, enc, dec, vocab, device):
    enc.eval()
    dec.eval()
    all_refs, all_preds = [], []

    rouge = Rouge()
    cider_scorer = Cider()

    with torch.no_grad():
        for feats, caps, cap_lens in tqdm(loader, desc="Evaluating"):
            feats, caps = feats.to(device), caps.to(device)
            B = feats.size(0)

            # Encoder Forward
            encoder_outs, _ = enc(feats)  # (B, T, enc_dim)

            # Initialize hidden states for BiLSTM
            num_directions = 2
            hidden_size = dec.bilstm.hidden_size
            num_layers = dec.bilstm.num_layers

            hidden = torch.zeros(num_layers * num_directions, B, hidden_size, device=device)
            cell = torch.zeros(num_layers * num_directions, B, hidden_size, device=device)


            #  Start decoding
            input_word = torch.LongTensor([vocab.word2idx[vocab.bos_token]] * B).to(device)
            preds = [[] for _ in range(B)]

            max_len = caps.size(1)

            for t in range(1, max_len):
                emb = dec.embedding(input_word).unsqueeze(1)  # (B,1,E)

                # Combine forward and backward hidden states from the last layer
                last_hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)  # (B, dec_hidden*2)

                # Attention over encoder outputs
                context, attn_weights = dec.attention(last_hidden_cat, encoder_outs)


                lstm_input = torch.cat([emb, context.unsqueeze(1)], dim=2)

                # One decoding step
                output, (hidden, cell) = dec.bilstm(lstm_input, (hidden, cell))
                out_vocab = dec.fc_out(dec.dropout(output.squeeze(1)))  # (B, vocab_size)

                # Greedy decoding
                top1 = out_vocab.argmax(1)
                input_word = top1

                for i in range(B):
                    preds[i].append(top1[i].item())


            #  Convert Predictions & References to Tokens
            for i in range(B):
                pred_tokens = []
                for tok in preds[i]:
                    if tok in (vocab.word2idx[vocab.pad_token], vocab.word2idx[vocab.bos_token]):
                        continue
                    if tok == vocab.word2idx[vocab.eos_token]:
                        break
                    pred_tokens.append(vocab.idx2word.get(tok, vocab.unk_token))
                all_preds.append(pred_tokens)

                ref_tokens = []
                for tok in caps[i].cpu().numpy():
                    if tok in (vocab.word2idx[vocab.pad_token], vocab.word2idx[vocab.bos_token]):
                        continue
                    if tok == vocab.word2idx[vocab.eos_token]:
                        break
                    ref_tokens.append(vocab.idx2word.get(int(tok), vocab.unk_token))
                all_refs.append([ref_tokens])


    #  Compute Evaluation Metrics
    smoothie = SmoothingFunction().method4
    bleu4 = corpus_bleu(all_refs, all_preds, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

    refs_str = [' '.join(ref[0]) for ref in all_refs]
    preds_str = [' '.join(pred) for pred in all_preds]

    rouge_scores = rouge.get_scores(preds_str, refs_str, avg=True)
    cider_score, _ = cider_scorer.compute_score(
        {i: [refs_str[i]] for i in range(len(refs_str))},
        {i: [preds_str[i]] for i in range(len(preds_str))}
    )

    print(f"\nBLEU-4: {bleu4:.4f}")
    print(f"ROUGE-L: {rouge_scores['rouge-l']['f']:.4f}")
    print(f"CIDEr: {cider_score:.4f}")

    return bleu4, rouge_scores, cider_score, all_refs, all_preds


# In[ ]:


def train_one_epoch(train_loader, enc, dec, optimizer, criterion, device, clip=5.0):
    enc.train()
    dec.train()
    running_loss = 0.0

    for feats, caps, cap_lens in tqdm(train_loader, desc="Training"):
        feats = feats.to(device)          # (B, T, D)
        caps = caps.to(device)            # (B, L)
        optimizer.zero_grad()


        # Encoder forward
        encoder_outs, _ = enc(feats)      # (B, T, enc_dim)


        # Decoder forward (BiLSTM)
        outputs, _ = dec(encoder_outs, caps, teacher_forcing_ratio=0.75)  # (B, L, V)


        # Ignore the <BOS> token
        outputs = outputs[:, 1:, :].contiguous()
        targets = caps[:, 1:].contiguous()

        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        loss.backward()

        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(dec.parameters()), clip)

        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(train_loader)


# In[ ]:


#Decder-BiLSTM 
#Training Loop

best_bleu = 0.0

for epoch in range(1, EPOCHS + 1):
    # Training 
    train_loss = train_one_epoch(train_loader, enc, dec, optimizer, criterion, DEVICE)
    print(f"Epoch {epoch} | Train Loss: {train_loss:.4f}")

    # Validation
    bleu, rouge_scores, cider, _, _ = evaluate_with_metrics(val_loader, enc, dec, vocab, DEVICE)
    rouge_l_f = rouge_scores['rouge-l']['f']

    print(f"[Epoch {epoch}] Validation Metrics:")
    print(f"  BLEU-4 = {bleu:.4f}")
    print(f"  ROUGE-L (F1) = {rouge_l_f:.4f}")
    print(f"  CIDEr = {cider:.4f}")


    if bleu > best_bleu:
        best_bleu = bleu
        torch.save({
            'enc_state': enc.state_dict(),
            'dec_state': dec.state_dict(),
            'vocab': vocab.word2idx
        }, "best_checkpoint.pth")
        print(" Saved new best checkpoint.")

print(f"\nTraining complete. Best BLEU-4 achieved: {best_bleu:.4f}")


# In[ ]:


def generate_caption_for_video(video_feat, enc, dec, vocab, device, max_len=30):
    enc.eval()
    dec.eval()
    with torch.no_grad():

        feat_tensor = torch.tensor(video_feat, dtype=torch.float32).unsqueeze(0).to(device)
        encoder_outs, _ = enc(feat_tensor)  # (1, T, enc_dim)

        batch_size = 1
        dec_hidden = dec.bilstm.hidden_size
        num_directions = 2  # BiLSTM
        enc_dim = encoder_outs.size(2)

        # Initialize hidden and cell states for BiLSTM
        hidden = torch.zeros(num_directions, batch_size, dec_hidden, device=device)
        cell = torch.zeros(num_directions, batch_size, dec_hidden, device=device)


        input_word = torch.tensor([vocab.word2idx[vocab.bos_token]], device=device)
        generated_words = []

        for _ in range(max_len):

            emb = dec.embedding(input_word).unsqueeze(1)  # (1,1,E)

            # Combine last forward & backward hidden states
            last_hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)  # (1, 2*dec_hidden)

            # Attention over encoder outputs
            context, _ = dec.attention(last_hidden_cat, encoder_outs)  # (1, enc_dim)

            # Prepare LSTM input (concat embedding + context)
            lstm_input = torch.cat([emb, context.unsqueeze(1)], dim=2)  # (1,1,E+enc_dim)

            # Run one BiLSTM step
            output, (hidden, cell) = dec.bilstm(lstm_input, (hidden, cell))  # output: (1,1,2*dec_hidden)


            out_vocab = dec.fc_out(dec.dropout(output.squeeze(1)))  # (1, vocab_size)


            top1 = out_vocab.argmax(1)
            word = vocab.idx2word.get(top1.item(), vocab.unk_token)

            # Stop if <EOS> token
            if word == vocab.eos_token:
                break

            generated_words.append(word)
            input_word = top1

        return ' '.join(generated_words)


# In[ ]:


import torch
import numpy as np
import os
import random
from tqdm import tqdm

#  Load the best saved checkpoint
ckpt = torch.load("/kaggle/input/bilstm-bestpath/best_checkpoint (5).pth", map_location=DEVICE)
enc.load_state_dict(ckpt['enc_state'])
dec.load_state_dict(ckpt['dec_state'])
print(" Loaded checkpoint successfully.")


#  Evaluate on test set
print("Running test evaluation (BLEU, ROUGE, CIDEr)...")
test_bleu, test_rouge, test_cider ,_,_= evaluate_with_metrics(test_loader, enc, dec, vocab, DEVICE)

test_rouge_l = test_rouge['rouge-l']['f']

print(f"\n📊 Test Metrics:")
print(f"BLEU-4  = {test_bleu:.4f}")
print(f"ROUGE-L = {test_rouge_l:.4f}")
print(f"CIDEr   = {test_cider:.4f}")


#  few random test samples

sample_videos = random.sample(list(test_items.keys()), 4)

print("\n --- Random Sample Predictions ---")
for i, vid in enumerate(sample_videos):
    sample_feat_path = os.path.join(FEATURES_DIR, vid.replace('.mp4', '.npy'))
    sample_feat = np.load(sample_feat_path)

    generated_caption = generate_caption_for_video(sample_feat, enc, dec, vocab, DEVICE)
    references = test_items[vid] 

    print(f"\n Video {i+1}: {vid}")
    print(f"Generated Caption: {generated_caption}")
    print("Reference Captions:")
    for j, ref in enumerate(references[:3]): 
        print(f"  Ref {j+1}: {ref}")


# In[ ]:


sample_videos = random.sample(list(test_items.keys()), 4)
selected_vids = [
    "video6643.mp4",
    "video1850.mp4",
    "video4829.mp4",
    "video8925.mp4"
]

print("\n --- Random Sample Predictions ---")
for i, vid in enumerate(selected_vids):
    sample_feat_path = os.path.join(FEATURES_DIR, vid.replace('.mp4', '.npy'))
    sample_feat = np.load(sample_feat_path)

    generated_caption = generate_caption_for_video(sample_feat, enc, dec, vocab, DEVICE)
    references = test_items[vid]  

    print(f"\n Video {i+1}: {vid}")
    print(f"Generated Caption: {generated_caption}")
    print("Reference Captions:")
    for j, ref in enumerate(references[:3]): 
        print(f"  Ref {j+1}: {ref}")

for i, vid in enumerate(sample_videos):
    sample_feat_path = os.path.join(FEATURES_DIR, vid.replace('.mp4', '.npy'))
    sample_feat = np.load(sample_feat_path)

    generated_caption = generate_caption_for_video(sample_feat, enc, dec, vocab, DEVICE)
    references = test_items[vid]  # list of ground-truth captions

    print(f"\n Video {i+1}: {vid}")
    print(f"Generated Caption: {generated_caption}")
    print("Reference Captions:")
    for j, ref in enumerate(references[:3]): 
        print(f"  Ref {j+1}: {ref}")


# LSTM-Encoder Tranformer-Decoder

# In[ ]:


import torch
import torch.nn as nn
import math


# LSTM Encoder

#no of layers=2 #bidirectional=true(1st)
class EncoderLSTM(nn.Module):
    def __init__(self, feat_dim=2048, hidden_dim=512, num_layers=1, bidirectional=True):
        super().__init__()
        self.lstm = nn.LSTM(feat_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=bidirectional)
        self.output_dim = hidden_dim * (2 if bidirectional else 1)

    def forward(self, feats):
        """
        feats: (B, T, feat_dim)
        Returns: (B, T, hidden_dim)
        """
        outputs, _ = self.lstm(feats)
        return outputs



# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)



# Transformer Decoder (No Cross Attention)

class TransformerDecoderNoAttn(nn.Module):
    def __init__(self, vocab_size, embed_dim, enc_dim, num_layers=3, ff_dim=1024, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        self.enc_proj = nn.Linear(enc_dim, embed_dim)  
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.embed_dim = embed_dim

    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.ones(sz, sz), diagonal=1).bool()

    def forward(self, encoder_outs, captions):

        B, T_dec = captions.size()
        tgt = self.embedding(captions) * math.sqrt(self.embed_dim)
        tgt = self.pos_encoder(tgt)

        tgt_mask = self.generate_square_subsequent_mask(T_dec).to(captions.device)
        memory = self.enc_proj(encoder_outs)

        out = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        out = self.fc_out(out)
        return out



# In[ ]:


#Parameters
FEATURE_DIM = 2048
ENC_HIDDEN = 512
EMBED_SIZE = 512
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

enc = EncoderLSTM(feat_dim=FEATURE_DIM, hidden_dim=ENC_HIDDEN).to(DEVICE)
dec = TransformerDecoderNoAttn(
    vocab_size=len(vocab.word2idx),
    embed_dim=EMBED_SIZE,
    enc_dim=enc.output_dim,
    num_layers=3
).to(DEVICE)

criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx[vocab.pad_token])
params = list(enc.parameters()) + list(dec.parameters())
optimizer = optim.Adam(params, lr=LR)



# Training loop
def train_one_epoch(train_loader, enc, dec, optimizer, criterion, device, clip=5.0):
    enc.train(); dec.train()
    running_loss = 0.0
    for feats, caps, cap_lens in tqdm(train_loader):
        feats, caps = feats.to(device), caps.to(device)
        optimizer.zero_grad()

        encoder_outs = enc(feats)
        outputs = dec(encoder_outs, caps[:, :-1])
        targets = caps[:, 1:]

        loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, clip)
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(train_loader)


# Evaluation (greedy decoding)
def evaluate(loader, enc, dec, device):
    enc.eval(); dec.eval()
    all_refs, all_preds = [], []

    with torch.no_grad():
        for feats, caps, cap_lens in tqdm(loader):
            feats, caps = feats.to(device), caps.to(device)
            encoder_outs = enc(feats)

            batch_size = feats.size(0)
            input_seq = torch.full((batch_size, 1),
                                   vocab.word2idx[vocab.bos_token],
                                   dtype=torch.long, device=device)
            preds = [[] for _ in range(batch_size)]

            for _ in range(caps.size(1)):
                outputs = dec(encoder_outs, input_seq)
                next_word = outputs[:, -1, :].argmax(1)
                input_seq = torch.cat([input_seq, next_word.unsqueeze(1)], dim=1)

                for i in range(batch_size):
                    preds[i].append(next_word[i].item())

            # Convert tokens to words
            for i in range(batch_size):
                pred_tokens = []
                for tok in preds[i]:
                    if tok == vocab.word2idx[vocab.eos_token]:
                        break
                    if tok in (vocab.word2idx[vocab.bos_token], vocab.word2idx[vocab.pad_token]):
                        continue
                    pred_tokens.append(vocab.idx2word.get(tok, vocab.unk_token))
                all_preds.append(pred_tokens)

                ref_tokens = []
                for tok in caps[i].cpu().numpy():
                    if tok == vocab.word2idx[vocab.eos_token]:
                        break
                    if tok in (vocab.word2idx[vocab.bos_token], vocab.word2idx[vocab.pad_token]):
                        continue
                    ref_tokens.append(vocab.idx2word.get(int(tok), vocab.unk_token))
                all_refs.append([ref_tokens])

    smoothie = SmoothingFunction().method4
    bleu4 = corpus_bleu(all_refs, all_preds, weights=(0.25,0.25,0.25,0.25), smoothing_function=smoothie)
    return bleu4, all_refs, all_preds


# In[ ]:


from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider
import numpy as np
import torch
from tqdm import tqdm

def evaluate_with_metrics(loader, enc, dec, vocab, device, max_len=20):
    enc.eval()
    dec.eval()
    all_refs = []
    all_preds = []

    with torch.no_grad():
        for feats, caps,cap_lens in tqdm(loader):
            feats = feats.to(device)
            caps = caps.to(device)

            # Encoder forward
            encoder_outs = enc(feats)  # (B, T_enc, enc_dim)

            # Greedy decoding
            batch_size = feats.size(0)
            input_word = torch.LongTensor([vocab.word2idx[vocab.bos_token]] * batch_size).to(device)
            preds = torch.zeros(batch_size, max_len).long().to(device)

            for t in range(max_len):
                out = dec(encoder_outs, input_word.unsqueeze(1))  # (B, 1, vocab_size)
                next_word = out[:, -1, :].argmax(-1)
                preds[:, t] = next_word
                input_word = next_word  

            # Convert preds and refs to token lists
            for i in range(batch_size):

                p = []
                for tok in preds[i].cpu().numpy():
                    if tok == vocab.word2idx[vocab.eos_token]:
                        break
                    if tok in (vocab.word2idx[vocab.pad_token], vocab.word2idx[vocab.bos_token]):
                        continue
                    p.append(vocab.idx2word.get(tok, vocab.unk_token))
                all_preds.append(p)

                # Reference tokens
                ref_tokens = []
                for tok in caps[i].cpu().numpy():
                    if tok == vocab.word2idx[vocab.eos_token]:
                        break
                    if tok in (vocab.word2idx[vocab.pad_token], vocab.word2idx[vocab.bos_token]):
                        continue
                    ref_tokens.append(vocab.idx2word.get(int(tok), vocab.unk_token))
                all_refs.append([ref_tokens]) 

    # Compute metrics
    smoothie = SmoothingFunction().method4
    bleu4 = corpus_bleu(all_refs, all_preds, weights=(0.25,0.25,0.25,0.25), smoothing_function=smoothie)


    # ROUGE-L
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l_f1s = [scorer.score(" ".join(ref[0]), " ".join(pred))['rougeL'].fmeasure for ref, pred in zip(all_refs, all_preds)]
    rouge_l = np.mean(rouge_l_f1s)

    # CIDEr

    cider_scorer = Cider()
    refs_for_cider = {i: [" ".join(r[0]) for r in all_refs[i:i+1]] for i in range(len(all_refs))}
    preds_for_cider = {i: [" ".join(all_preds[i])] for i in range(len(all_preds))}  
    cider_score, _ = cider_scorer.compute_score(refs_for_cider, preds_for_cider)


    return bleu4 ,rouge_l, cider_score


# In[54]:


best_bleu = 0.0
EPOCHS = 10

for epoch in range(1, EPOCHS+1):
    # Train
    train_loss = train_one_epoch(train_loader, enc, dec, optimizer, criterion, DEVICE)
    print(f"[Epoch {epoch}] Train loss: {train_loss:.4f}")

    # Validate with metrics
    bleu, rouge_l, cider,_,_ = evaluate_with_metrics(val_loader, enc, dec,vocab, DEVICE)
    print(f"[Epoch {epoch}] Validation Metrics: BLEU-4={bleu:.4f}, ROUGE-L={rouge_l:.4f}, CIDEr={cider:.4f}")

    # Save best BLEU model
    if bleu > best_bleu:
        best_bleu = bleu
        torch.save({
            'enc_state': enc.state_dict(),
            'dec_state': dec.state_dict(),
            'vocab': vocab.word2idx
        }, "best_checkpoint_metrics.pth")
        print(f"✅ Saved new best model at epoch {epoch} with BLEU={bleu:.4f}")


# In[ ]:


#biLSTMEnc(1L)+Tranformer
best_bleu = 0.0
EPOCHS = 10

for epoch in range(1, EPOCHS+1):
    # Train
    train_loss = train_one_epoch(train_loader, enc, dec, optimizer, criterion, DEVICE)
    print(f"[Epoch {epoch}] Train loss: {train_loss:.4f}")

    # Validate with metrics
    bleu, rouge_l, cider = evaluate_with_metrics(val_loader, enc, dec,vocab, DEVICE)
    print(f"[Epoch {epoch}] Validation Metrics: BLEU-4={bleu:.4f}, ROUGE-L={rouge_l:.4f}, CIDEr={cider:.4f}")

    # Save best BLEU model
    if bleu > best_bleu:
        best_bleu = bleu
        torch.save({
            'enc_state': enc.state_dict(),
            'dec_state': dec.state_dict(),
            'vocab': vocab.word2idx
        }, "best_checkpoint_metrics.pth")
        print(f"✅ Saved new best model at epoch {epoch} with BLEU={bleu:.4f}")


# In[16]:


#bidrectional LSTM +Transformer=true
best_bleu = 0.0
EPOCHS = 10

for epoch in range(1, EPOCHS+1):
    # Train
    train_loss = train_one_epoch(train_loader, enc, dec, optimizer, criterion, DEVICE)
    print(f"[Epoch {epoch}] Train loss: {train_loss:.4f}")

    # Validate with metrics
    bleu, rouge_l, cider = evaluate_with_metrics(val_loader, enc, dec,vocab, DEVICE)
    print(f"[Epoch {epoch}] Validation Metrics: BLEU-4={bleu:.4f}, ROUGE-L={rouge_l:.4f}, CIDEr={cider:.4f}")

    # Save best BLEU model
    if bleu > best_bleu:
        best_bleu = bleu
        torch.save({
            'enc_state': enc.state_dict(),
            'dec_state': dec.state_dict(),
            'vocab': vocab.word2idx
        }, "best_checkpoint_metrics.pth")
        print(f"✅ Saved new best model at epoch {epoch} with BLEU={bleu:.4f}")


# In[ ]:


import torch

# Load best model
checkpoint = torch.load("best_checkpoint_metrics.pth", map_location=DEVICE)

enc.load_state_dict(checkpoint['enc_state'])
dec.load_state_dict(checkpoint['dec_state'])
print(" Loaded best model checkpoint successfully.")

enc.eval()
dec.eval()


bleu, rouge_l, cider = evaluate_with_metrics(test_loader, enc, dec, vocab, DEVICE)

print("\n📊 Final Test Results:")
print(f"BLEU-4  : {bleu:.4f}")
print(f"ROUGE-L : {rouge_l:.4f}")
print(f"CIDEr   : {cider:.4f}")


# In[ ]:


#With 2 layer in encoder bilstm 
best_bleu = 0.0
EPOCHS = 10


for epoch in range(1, EPOCHS+1):
    # Train
    train_loss = train_one_epoch(train_loader, enc, dec, optimizer, criterion, DEVICE)
    print(f"[Epoch {epoch}] Train loss: {train_loss:.4f}")

    # Validate with metrics
    bleu, rouge_l, cider = evaluate_with_metrics(val_loader, enc, dec,vocab, DEVICE)
    print(f"[Epoch {epoch}] Validation Metrics: BLEU-4={bleu:.4f}, ROUGE-L={rouge_l:.4f}, CIDEr={cider:.4f}")

    if bleu > best_bleu:
        best_bleu = bleu
        torch.save({
            'enc_state': enc.state_dict(),
            'dec_state': dec.state_dict(),
            'vocab': vocab.word2idx
        }, "best_checkpoint_metrics.pth")
        print(f" Saved new best model at epoch {epoch} with BLEU={bleu:.4f}")


# In[ ]:


import torch
import numpy as np
import os
import random
from tqdm import tqdm

#Load the best saved checkpoint
ckpt = torch.load("best_checkpoint_metrics.pth", map_location=DEVICE)
enc.load_state_dict(ckpt['enc_state'])
dec.load_state_dict(ckpt['dec_state'])
print(" Loaded checkpoint successfully.")

#  Evaluate on test set
print("Running test evaluation (BLEU, ROUGE, CIDEr)...")
test_bleu, test_rouge, test_cider = evaluate_with_metrics(test_loader, enc, dec, vocab, DEVICE)
print(f"\n📊 Test Metrics:\nBLEU-4 = {test_bleu:.4f}\nROUGE-L = {test_rouge:.4f}\nCIDEr = {test_cider:.4f}")


def generate_caption_for_video(video_feat, enc, dec, vocab, device, max_len=20):
    enc.eval()
    dec.eval()

    feat_tensor = torch.tensor(video_feat).unsqueeze(0).float().to(device)  # (1, T, D)
    with torch.no_grad():
        enc_outs = enc(feat_tensor)


        input_seq = torch.LongTensor([[vocab.word2idx[vocab.bos_token]]]).to(device)
        generated_tokens = []

        for _ in range(max_len):
            out = dec(enc_outs, input_seq)
            next_word = out[:, -1, :].argmax(-1).item()
            if next_word == vocab.word2idx[vocab.eos_token]:
                break
            generated_tokens.append(vocab.idx2word.get(next_word, vocab.unk_token))
            input_seq = torch.cat([input_seq, torch.LongTensor([[next_word]]).to(device)], dim=1)

    return " ".join(generated_tokens)


sample_videos = random.sample(list(test_items.keys()), 4)

print("\n --- Random Sample Predictions ---")
for i, vid in enumerate(sample_videos):
    sample_feat_path = os.path.join(FEATURES_DIR, vid.replace('.mp4', '.npy'))
    sample_feat = np.load(sample_feat_path)

    generated_caption = generate_caption_for_video(sample_feat, enc, dec, vocab, DEVICE)
    references = test_items[vid] 

    print(f"\n Video {i+1}: {vid}")
    print(f"Generated Caption: {generated_caption}")
    print("Reference Captions:")
    for j, ref in enumerate(references[:3]):  
        print(f"  Ref {j+1}: {ref}")


# In[ ]:


import torch
import numpy as np
import os
import random
from tqdm import tqdm

#  Load the best saved checkpoint
ckpt = torch.load("best_checkpoint_metrics.pth", map_location=DEVICE)
enc.load_state_dict(ckpt['enc_state'])
dec.load_state_dict(ckpt['dec_state'])
print(" Loaded checkpoint successfully.")



#  Evaluate on test set
print("Running test evaluation (BLEU, ROUGE, CIDEr)...")
test_bleu, test_rouge, test_cider = evaluate_with_metrics(test_loader, enc, dec, vocab, DEVICE)
print(f"\n📊 Test Metrics:\nBLEU-4 = {test_bleu:.4f}\nROUGE-L = {test_rouge:.4f}\nCIDEr = {test_cider:.4f}")



def generate_caption_for_video(video_feat, enc, dec, vocab, device, max_len=20):
    enc.eval()
    dec.eval()

    feat_tensor = torch.tensor(video_feat).unsqueeze(0).float().to(device)  # (1, T, D)
    with torch.no_grad():
        enc_outs = enc(feat_tensor)


        input_seq = torch.LongTensor([[vocab.word2idx[vocab.bos_token]]]).to(device)
        generated_tokens = []

        for _ in range(max_len):
            out = dec(enc_outs, input_seq)
            next_word = out[:, -1, :].argmax(-1).item()
            if next_word == vocab.word2idx[vocab.eos_token]:
                break
            generated_tokens.append(vocab.idx2word.get(next_word, vocab.unk_token))
            input_seq = torch.cat([input_seq, torch.LongTensor([[next_word]]).to(device)], dim=1)

    return " ".join(generated_tokens)

def predict_for_selected_videos(selected_vids, features_dir, test_items, enc, dec, vocab, device, max_len=20):

    print("\n--- Selected Video Predictions ---")
    for i, vid in enumerate(selected_vids):
        feat_path = os.path.join(features_dir, vid.replace('.mp4', '.npy'))
        if not os.path.exists(feat_path):
            print(f"⚠️ Feature file missing for {vid}")
            continue

        video_feat = np.load(feat_path)
        generated_caption = generate_caption_for_video(video_feat, enc, dec, vocab, device, max_len)
        references = test_items.get(vid, [])

        print(f"\nVideo {i+1}: {vid}")
        print(f"Generated Caption: {generated_caption}")
        print("Reference Captions:")
        for j, ref in enumerate(references[:3]):  # show up to 3 refs
            print(f"  Ref {j+1}: {ref}")

# Example: Choose specific video IDs from your test set
selected_vids = [
    "video6643.mp4",
    "video1850.mp4",
    "video4829.mp4",
    "video8925.mp4"
]

predict_for_selected_videos(
    selected_vids,
    FEATURES_DIR,
    test_items,
    enc,
    dec,
    vocab,
    DEVICE,
    max_len=20
)

# few random test samples

sample_videos = random.sample(list(test_items.keys()), 4)

print("\n --- Random Sample Predictions ---")
for i, vid in enumerate(sample_videos):
    sample_feat_path = os.path.join(FEATURES_DIR, vid.replace('.mp4', '.npy'))
    sample_feat = np.load(sample_feat_path)

    generated_caption = generate_caption_for_video(sample_feat, enc, dec, vocab, DEVICE)
    references = test_items[vid]

    print(f"\n Video {i+1}: {vid}")
    print(f"Generated Caption: {generated_caption}")
    print("Reference Captions:")
    for j, ref in enumerate(references[:3]):  
        print(f"  Ref {j+1}: {ref}")


# Comparison between two Decoders used:

# In[2]:


import matplotlib.pyplot as plt

models = ['LSTM+Luong', 'LSTM+Transformer']
bleu = [0.0535, 0.0142]
rouge = [0.281, 0.2028]
cider = [0.5009, 0.2119]

plt.figure(figsize=(8, 5))
plt.plot(models, bleu, marker='o', label='BLEU-4')
plt.plot(models, rouge, marker='s', label='ROUGE-L')
plt.plot(models, cider, marker='^', label='CIDEr')
plt.xlabel('Model')
plt.ylabel('Score')
plt.title('Video Caption Generation Metric Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:


#bidirectional=True +transformer
import torch
import numpy as np
import os
import random
from tqdm import tqdm

#  Load the best saved checkpoint

ckpt = torch.load("best_checkpoint_metrics.pth", map_location=DEVICE)
enc.load_state_dict(ckpt['enc_state'])
dec.load_state_dict(ckpt['dec_state'])
print(" Loaded checkpoint successfully.")


#Evaluate on test set

print("Running test evaluation (BLEU, ROUGE, CIDEr)...")
test_bleu, test_rouge, test_cider = evaluate_with_metrics(test_loader, enc, dec, vocab, DEVICE)
print(f"\n📊 Test Metrics:\nBLEU-4 = {test_bleu:.4f}\nROUGE-L = {test_rouge:.4f}\nCIDEr = {test_cider:.4f}")

#  Function to generate caption for a single video

def generate_caption_for_video(video_feat, enc, dec, vocab, device, max_len=20):
    enc.eval()
    dec.eval()

    feat_tensor = torch.tensor(video_feat).unsqueeze(0).float().to(device)  # (1, T, D)
    with torch.no_grad():
        enc_outs = enc(feat_tensor)


        input_seq = torch.LongTensor([[vocab.word2idx[vocab.bos_token]]]).to(device)
        generated_tokens = []

        for _ in range(max_len):
            out = dec(enc_outs, input_seq)
            next_word = out[:, -1, :].argmax(-1).item()
            if next_word == vocab.word2idx[vocab.eos_token]:
                break
            generated_tokens.append(vocab.idx2word.get(next_word, vocab.unk_token))
            input_seq = torch.cat([input_seq, torch.LongTensor([[next_word]]).to(device)], dim=1)

    return " ".join(generated_tokens)

def predict_for_selected_videos(selected_vids, features_dir, test_items, enc, dec, vocab, device, max_len=20):

    print("\n--- Selected Video Predictions ---")
    for i, vid in enumerate(selected_vids):
        feat_path = os.path.join(features_dir, vid.replace('.mp4', '.npy'))
        if not os.path.exists(feat_path):
            print(f" Feature file missing for {vid}")
            continue

        video_feat = np.load(feat_path)
        generated_caption = generate_caption_for_video(video_feat, enc, dec, vocab, device, max_len)
        references = test_items.get(vid, [])

        print(f"\nVideo {i+1}: {vid}")
        print(f"Generated Caption: {generated_caption}")
        print("Reference Captions:")
        for j, ref in enumerate(references[:3]):  # show up to 3 refs
            print(f"  Ref {j+1}: {ref}")

# Example: Choose specific video IDs from your test set
selected_vids = [
    "video6643.mp4",
    "video1850.mp4",
    "video4829.mp4",
    "video8925.mp4"
]

predict_for_selected_videos(
    selected_vids,
    FEATURES_DIR,
    test_items,
    enc,
    dec,
    vocab,
    DEVICE,
    max_len=20
)

# =====================================
# 🔹 Pick a few random test samples
# =====================================
sample_videos = random.sample(list(test_items.keys()), 4)

print("\n --- Random Sample Predictions ---")
for i, vid in enumerate(sample_videos):
    sample_feat_path = os.path.join(FEATURES_DIR, vid.replace('.mp4', '.npy'))
    sample_feat = np.load(sample_feat_path)

    generated_caption = generate_caption_for_video(sample_feat, enc, dec, vocab, DEVICE)
    references = test_items[vid]  # list of ground-truth captions

    print(f"\n Video {i+1}: {vid}")
    print(f"Generated Caption: {generated_caption}")
    print("Reference Captions:")
    for j, ref in enumerate(references[:3]):  # print up to 3 refs
        print(f"  Ref {j+1}: {ref}")


# Other experiments done: Spatial+temporal attention:

# In[34]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialTemporalAttention(nn.Module):
    """
    Add lightweight spatial + temporal attention between encoder and decoder.
    Works with (B, T, D_enc) encoder outputs.
    """
    def __init__(self, enc_dim, use_spatial=False):
        super().__init__()
        self.enc_dim = enc_dim
        self.use_spatial = use_spatial

        # Temporal attention: learns to weight frames
        self.temporal_attn = nn.Sequential(
            nn.Linear(enc_dim, enc_dim // 2),
            nn.Tanh(),
            nn.Linear(enc_dim // 2, 1)
        )


        if use_spatial:
            self.spatial_attn = nn.Sequential(
                nn.Linear(enc_dim, enc_dim // 2),
                nn.Tanh(),
                nn.Linear(enc_dim // 2, 1)
            )

    def forward(self, enc_outs, spatial_feats=None):
        """
        enc_outs: (B, T, D_enc)
        spatial_feats (optional): (B, T, R, D_enc)
        """
        # 1️ Spatial attention (optional)
        if self.use_spatial and spatial_feats is not None:
            attn_weights = F.softmax(self.spatial_attn(spatial_feats).squeeze(-1), dim=-1)  # (B,T,R)
            spatial_context = (attn_weights.unsqueeze(-1) * spatial_feats).sum(dim=2)      # (B,T,D)
            enc_outs = enc_outs + spatial_context  # residual fusion

        # 2️ Temporal attention
        energy = self.temporal_attn(enc_outs).squeeze(-1)      # (B,T)
        weights = F.softmax(energy, dim=-1).unsqueeze(-1)      # (B,T,1)
        attended = (weights * enc_outs).sum(dim=1, keepdim=True)  # (B,1,D)

        # Optionally repeat context for decoder
        refined_outs = attended.repeat(1, enc_outs.size(1), 1)
        return refined_outs


# In[35]:


FEATURE_DIM = 2048
ENC_HIDDEN = 512
EMBED_SIZE = 512
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

enc = EncoderLSTM(feat_dim=FEATURE_DIM, hidden_dim=ENC_HIDDEN).to(DEVICE)
dec = TransformerDecoderNoAttn(
    vocab_size=len(vocab.word2idx),
    embed_dim=EMBED_SIZE,
    enc_dim=enc.output_dim,
    num_layers=3
).to(DEVICE)

# ✅ Add the attention bridge
attn_refiner = SpatialTemporalAttention(enc_dim=enc.output_dim).to(DEVICE)

criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx[vocab.pad_token])
params = list(enc.parameters()) + list(dec.parameters()) + list(attn_refiner.parameters())
optimizer = optim.Adam(params, lr=LR)


# In[36]:


def train_one_epoch(train_loader, enc, dec, attn_refiner, optimizer, criterion, device, clip=5.0):
    enc.train(); dec.train(); attn_refiner.train()
    running_loss = 0.0

    for feats, caps, cap_lens in tqdm(train_loader):
        feats, caps = feats.to(device), caps.to(device)
        optimizer.zero_grad()

        encoder_outs = enc(feats)                        # (B, T, D_enc)
        refined_enc_outs = attn_refiner(encoder_outs)    # (B, T, D_enc) — attended features

        outputs = dec(refined_enc_outs, caps[:, :-1])
        targets = caps[:, 1:]

        loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, clip)
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)


# In[87]:


best_bleu = 0.0
EPOCHS = 10

for epoch in range(1, EPOCHS+1):
    # Train
    train_loss = train_one_epoch(train_loader, enc, dec, attn_refiner, optimizer, criterion, DEVICE)
    print(f"[Epoch {epoch}] Train loss: {train_loss:.4f}")

    # Validate with metrics
    bleu, rouge_l, cider = evaluate_with_metrics(val_loader, enc, dec,attn_refiner, vocab, DEVICE)
    print(f"[Epoch {epoch}] Validation Metrics: BLEU-4={bleu:.4f}, ROUGE-L={rouge_l:.4f}, CIDEr={cider:.4f}")

    # Save best BLEU model
    if bleu > best_bleu:
        best_bleu = bleu
        torch.save({
            'enc_state': enc.state_dict(),
            'dec_state': dec.state_dict(),
            'vocab': vocab.word2idx
        }, "best_checkpoint_metrics.pth")
        print(f" Saved new best model at epoch {epoch} with BLEU={bleu:.4f}")


# In[ ]:


from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider
import numpy as np
import torch
from tqdm import tqdm

def evaluate_with_metrics(loader, enc, dec, attn_refiner,vocab, device, max_len=20):
    enc.eval()
    dec.eval()
    all_refs = []
    all_preds = []

    with torch.no_grad():
        for feats, caps,cap_lens in tqdm(loader):
            feats = feats.to(device)
            caps = caps.to(device)

            # Encoder forward
            encoder_outs = enc(feats)  # (B, T_enc, enc_dim)

            encoder_outs = attn_refiner(encoder_outs)

            # Greedy decoding
            batch_size = feats.size(0)
            input_word = torch.LongTensor([vocab.word2idx[vocab.bos_token]] * batch_size).to(device)
            preds = torch.zeros(batch_size, max_len).long().to(device)

            for t in range(max_len):
                out = dec(encoder_outs, input_word.unsqueeze(1))  # (B, 1, vocab_size)
                next_word = out[:, -1, :].argmax(-1)
                preds[:, t] = next_word
                input_word = next_word 

            # Convert preds and refs to token lists
            for i in range(batch_size):

                p = []
                for tok in preds[i].cpu().numpy():
                    if tok == vocab.word2idx[vocab.eos_token]:
                        break
                    if tok in (vocab.word2idx[vocab.pad_token], vocab.word2idx[vocab.bos_token]):
                        continue
                    p.append(vocab.idx2word.get(tok, vocab.unk_token))
                all_preds.append(p)

                # Reference tokens
                ref_tokens = []
                for tok in caps[i].cpu().numpy():
                    if tok == vocab.word2idx[vocab.eos_token]:
                        break
                    if tok in (vocab.word2idx[vocab.pad_token], vocab.word2idx[vocab.bos_token]):
                        continue
                    ref_tokens.append(vocab.idx2word.get(int(tok), vocab.unk_token))
                all_refs.append([ref_tokens])  


    # Compute metrics
    smoothie = SmoothingFunction().method4
    bleu4 = corpus_bleu(all_refs, all_preds, weights=(0.25,0.25,0.25,0.25), smoothing_function=smoothie)



    # ROUGE-L
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l_f1s = [scorer.score(" ".join(ref[0]), " ".join(pred))['rougeL'].fmeasure for ref, pred in zip(all_refs, all_preds)]
    rouge_l = np.mean(rouge_l_f1s)

  #CidER
    cider_scorer = Cider()
    refs_for_cider = {i: [" ".join(r[0]) for r in all_refs[i:i+1]] for i in range(len(all_refs))}
    preds_for_cider = {i: [" ".join(all_preds[i])] for i in range(len(all_preds))} 
    cider_score, _ = cider_scorer.compute_score(refs_for_cider, preds_for_cider)


    return bleu4 ,rouge_l, cider_score


# In[ ]:


import torch
import numpy as np
import os
import random
from tqdm import tqdm


#  Load the best saved checkpoint
ckpt = torch.load("best_checkpoint_metrics.pth", map_location=DEVICE)
enc.load_state_dict(ckpt['enc_state'])
dec.load_state_dict(ckpt['dec_state'])
print(" Loaded checkpoint successfully.")



#  Evaluate on test set
print("Running test evaluation (BLEU, ROUGE, CIDEr)...")
test_bleu, test_rouge, test_cider = evaluate_with_metrics(test_loader, enc, dec, vocab, DEVICE)
print(f"\n📊 Test Metrics:\nBLEU-4 = {test_bleu:.4f}\nROUGE-L = {test_rouge:.4f}\nCIDEr = {test_cider:.4f}")



def generate_caption_for_video(video_feat, enc, dec, vocab, device, max_len=20):
    enc.eval()
    dec.eval()

    feat_tensor = torch.tensor(video_feat).unsqueeze(0).float().to(device)  # (1, T, D)
    with torch.no_grad():
        enc_outs = enc(feat_tensor)


        input_seq = torch.LongTensor([[vocab.word2idx[vocab.bos_token]]]).to(device)
        generated_tokens = []

        for _ in range(max_len):
            out = dec(enc_outs, input_seq)
            next_word = out[:, -1, :].argmax(-1).item()
            if next_word == vocab.word2idx[vocab.eos_token]:
                break
            generated_tokens.append(vocab.idx2word.get(next_word, vocab.unk_token))
            input_seq = torch.cat([input_seq, torch.LongTensor([[next_word]]).to(device)], dim=1)

    return " ".join(generated_tokens)

# few random test samples
sample_videos = random.sample(list(test_items.keys()), 4)

print("\n --- Random Sample Predictions ---")
for i, vid in enumerate(sample_videos):
    sample_feat_path = os.path.join(FEATURES_DIR, vid.replace('.mp4', '.npy'))
    sample_feat = np.load(sample_feat_path)

    generated_caption = generate_caption_for_video(sample_feat, enc, dec, vocab, DEVICE)
    references = test_items[vid] 

    print(f"\n Video {i+1}: {vid}")
    print(f"Generated Caption: {generated_caption}")
    print("Reference Captions:")
    for j, ref in enumerate(references[:3]): 
        print(f"  Ref {j+1}: {ref}")

