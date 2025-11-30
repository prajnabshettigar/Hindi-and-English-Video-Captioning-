#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[1]:


# Load and check dimensions of Global Features
import os
import numpy as np


folder_path = "/kaggle/input/msr-vtt-1289-hindi-english/features_global/features_global"


npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]


if not npy_files:
    print("No .npy files found in the folder.")
else:

    file_path = os.path.join(folder_path, npy_files[0])
    print(f"Checking dimensions of: {file_path}")


    data = np.load(file_path)
    print("Shape:", data.shape)
    print("Data type:", data.dtype)


# In[2]:


# Load and check dimensions of local Features
import os
import numpy as np


folder_path = "/kaggle/input/msr-vtt-1289-hindi-english/features_local/features_local"

npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]


if not npy_files:
    print("No .npy files found in the folder.")
else:

    file_path = os.path.join(folder_path, npy_files[0])
    print(f"Checking dimensions of: {file_path}")


    data = np.load(file_path)
    print("Shape:", data.shape)
    print("Data type:", data.dtype)


# In[4]:


# Load and check dimensions of motion Features
import os
import numpy as np


folder_path = "/kaggle/input/msr-vtt-1289-hindi-english/features_motion/features_motion"


npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]


if not npy_files:
    print("No .npy files found in the folder.")
else:

    file_path = os.path.join(folder_path, npy_files[0])
    print(f"Checking dimensions of: {file_path}")


    data = np.load(file_path)
    print("Shape:", data.shape)
    print("Data type:", data.dtype)


# In[1]:


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


# In[42]:


#Parameters for BiLSTM Encoder-LSTM Decoder model
SAMPLE_FRAMES = 16
FEATURE_DIM   = 4608   # ← 2048 + 2048 + 512
ENC_HIDDEN    = 512
DEC_HIDDEN    = 512
EMBED_SIZE    = 512
BATCH_SIZE    = 32
LR            = 1e-4
EPOCHS        = 15
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)


# In[15]:


import json

#  captions.json file with english captions for 10000 videos(20 for each)
CAPTIONS_JSON = "/kaggle/input/msr-vtt-1289-hindi-english/captions.json"

# Load JSON file
with open(CAPTIONS_JSON, "r") as f:
    captions = json.load(f)


video_ids = list(captions.keys())


unique_videos = len(set(video_ids))

print(f"🔹 Total entries in JSON: {len(video_ids)}")
print(f"🔹 Unique video IDs: {unique_videos}")


# In[2]:


#  vocabulary from captions for English Caption Generation
from nltk.tokenize import word_tokenize

CAPTIONS_FILE = os.path.join("/kaggle/input/msr-vtt-1289-hindi-english/captions.json") 

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


# In[3]:


#Selecting first 1290 videos in order
import json

CAPTIONS_JSON = "/kaggle/input/msr-vtt-1289-hindi-english/captions.json"

# Load the captions
with open(CAPTIONS_JSON, "r") as f:
    captions = json.load(f)

print("Total videos in captions.json:", len(captions))

# Function to extract number from 'video1234.mp4'
def get_video_number(vid_name):
    return int(''.join(ch for ch in vid_name if ch.isdigit()))

# Sort videos numerically by number
sorted_videos = sorted(captions.keys(), key=get_video_number)

# Keep first 1289 videos
captions_filtered = {vid: captions[vid] for vid in sorted_videos[:1290]}

print("Videos after filtering:", len(captions_filtered))
print("First few video IDs:", list(captions_filtered.keys())[:10])
print("Last few video IDs:", list(captions_filtered.keys())[-10:])


# **Loading the dataset into Dataloager and forming Train,validation and test data**

# In[20]:


import os
import torch
import numpy as np
from torch.utils.data import Dataset
import random

FEATURES_GLOBAL_DIR = "/kaggle/input/msr-vtt-1289-hindi-english/features_global/features_global"
FEATURES_LOCAL_DIR  = "/kaggle/input/msr-vtt-1289-hindi-english/features_local/features_local"
FEATURES_MOTION_DIR = "/kaggle/input/msr-vtt-1289-hindi-english/features_motion/features_motion"

class MSRVTTMultiFeatureDataset(Dataset):
    def __init__(self, captions_dict, vocab, sample_frames=16, max_caption_len=30):
        self.items = [(vid, c) for vid, caps in captions_dict.items() for c in caps]
        self.vocab = vocab
        self.sample_frames = sample_frames
        self.max_caption_len = max_caption_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        vid, cap = self.items[idx]
        vid_id = vid.replace(".mp4", ".npy")


        global_path = os.path.join(FEATURES_GLOBAL_DIR, vid_id)
        local_path  = os.path.join(FEATURES_LOCAL_DIR, vid_id)
        motion_path = os.path.join(FEATURES_MOTION_DIR, vid_id)

        global_feat = np.load(global_path)   # (T, 2048)
        local_feat  = np.load(local_path)    # (T, 49, 2048)
        motion_feat = np.load(motion_path)   # (512,)



        #  Fix unexpected shapes 
        if global_feat.ndim == 1:
            global_feat = np.expand_dims(global_feat, axis=0)

        if local_feat.ndim == 3:
            local_mean = local_feat.mean(axis=1)  # (T, 2048)
        elif local_feat.ndim == 2:
            local_mean = local_feat               # already (T, 2048)
        elif local_feat.ndim == 1:
            local_mean = np.expand_dims(local_feat, axis=0)  # (1, D)
        else:
            raise ValueError(f"Unexpected local_feat shape: {local_feat.shape}")

        if motion_feat.ndim == 1:
            motion_repeat = np.repeat(motion_feat[np.newaxis, :], global_feat.shape[0], axis=0)
        else:
            raise ValueError(f"Unexpected motion_feat shape: {motion_feat.shape}")



        #Concatenate along feature dimension
        feats = np.concatenate([global_feat, local_mean, motion_repeat], axis=1)  # (T, 4608)

        #Frame padding/truncation
        if feats.shape[0] < self.sample_frames:
            pad = np.zeros((self.sample_frames - feats.shape[0], feats.shape[1]), dtype=np.float32)
            feats = np.concatenate([feats, pad], axis=0)
        else:
            feats = feats[:self.sample_frames]

        # Caption numericalization + padding
        numer = self.vocab.numericalize(cap)
        if len(numer) > self.max_caption_len:
            numer = numer[:self.max_caption_len-1] + [self.vocab.word2idx[self.vocab.eos_token]]

        cap_len = len(numer)
        pad_len = self.max_caption_len - cap_len
        if pad_len > 0:
            numer = numer + [self.vocab.word2idx[self.vocab.pad_token]] * pad_len

        return torch.FloatTensor(feats), torch.LongTensor(numer), cap_len


def collate_fn(batch):
    feats = torch.stack([b[0] for b in batch], dim=0)  # (B, T, D)
    caps = torch.stack([b[1] for b in batch], dim=0)
    cap_lens = torch.LongTensor([b[2] for b in batch])
    return feats, caps, cap_lens

items = list(captions_hindi.items())
random.seed(42)
random.shuffle(items)
n = len(items)
train_items = dict(items[:int(0.8*n)])
val_items = dict(items[int(0.8*n):int(0.9*n)])
test_items = dict(items[int(0.9*n):])

train_ds = MSRVTTMultiFeatureDataset(train_items, vocab)
val_ds   = MSRVTTMultiFeatureDataset(val_items, vocab)
test_ds  = MSRVTTMultiFeatureDataset(test_items, vocab)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4)

print("Train size:", len(train_ds), "Val size:", len(val_ds), "Test size:", len(test_ds))


# **BiLSTM Encoder+ LSTM Decoder with Luong attention**

# In[43]:


# Model-1: BiLSTM Encoder+ Decoder with Luong attention

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

        # project encoder outputs to decoder dim
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
        # use last_hidden as query
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

        # Initialize hidden state and cell to zeros
        hidden = torch.zeros(batch_size, self.lstm.hidden_size, device=encoder_outputs.device)
        cell = torch.zeros(batch_size, self.lstm.hidden_size, device=encoder_outputs.device)

        outputs = torch.zeros(batch_size, max_len, vocab_size, device=encoder_outputs.device)
        attn_weights_all = []

        # first input is <BOS>
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


# **BiLSTM encoder+Transforme decoder with only self attention**

# In[7]:


import torch
import torch.nn as nn
import math

# ----------------------------
# LSTM Encoder
# ----------------------------
#no of layers=2 #bidirectional=true(1st)- tried with layers 1,2,3,5,4 etc.

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

# ----------------------------
# Positional Encoding
# ----------------------------
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


# ----------------------------
# Transformer Decoder (No Cross Attention)
# ----------------------------
class TransformerDecoderNoAttn(nn.Module):
    def __init__(self, vocab_size, embed_dim, enc_dim, num_layers=3, ff_dim=1024, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        self.enc_proj = nn.Linear(enc_dim, embed_dim)  # project LSTM outputs
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
        # encoder_outs: (B, T_enc, enc_dim)
        # captions: (B, T_dec)
        B, T_dec = captions.size()
        tgt = self.embedding(captions) * math.sqrt(self.embed_dim)
        tgt = self.pos_encoder(tgt)

        tgt_mask = self.generate_square_subsequent_mask(T_dec).to(captions.device)
        memory = self.enc_proj(encoder_outs)

        out = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        out = self.fc_out(out)
        return out



# **BiLSTM Encoder+TRANSFORMER( WITH CROSS ATTENTION)**

# In[6]:


import torch
import torch.nn as nn
import math
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

class TransformerDecoderWithCrossAttn(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        enc_dim: int,
        num_layers: int = 3,
        nhead: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        max_len: int = 500
    ):
        super().__init__()
        assert embed_dim % nhead == 0, "embed_dim must be divisible by nhead"
        self.embed_dim = embed_dim

        # token embedding + positional encoding
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout, max_len=max_len)

        # project encoder outputs to decoder d_model
        self.enc_proj = nn.Linear(enc_dim, embed_dim)

        # transformer decoder (uses cross-attention internally)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True

        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # final linear -> vocab
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def _generate_square_subsequent_mask(self, sz, device):
        # boolean mask where True means masked (no attention)
        return torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()

    def forward(self, encoder_outs, captions, tgt_mask=None, memory_key_padding_mask=None):

        device = captions.device
        B, T_dec = captions.size()

        # embed tokens and add positional encodings
        tgt = self.embedding(captions) * math.sqrt(self.embed_dim)  # (B, T_dec, embed_dim)
        tgt = self.pos_encoder(tgt)

        # project encoder outputs to embed_dim
        memory = self.enc_proj(encoder_outs)  # (B, T_enc, embed_dim)

        # causal mask if not provided so that mode wont look ahead
        if tgt_mask is None:
            tgt_mask = self._generate_square_subsequent_mask(T_dec, device=device)  # bool mask

        # Pass memory_key_padding_mask to ignore padded encoder positions (True means pad).
        out = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_key_padding_mask=memory_key_padding_mask)

        logits = self.fc_out(out)  # (B, T_dec, vocab_size)
        return logits



# **Experiments**

# In[ ]:


1)BILSTM(1 layer+BiLSTM)+decoder(LSTM)
enc = EncoderRNN(feat_size=FEATURE_DIM, hidden_size=ENC_HIDDEN, bidirectional=True).to(DEVICE)
dec = DecoderWithAttention(embed_size=EMBED_SIZE, enc_dim=enc.output_size, dec_hidden=DEC_HIDDEN, vocab_size=len(vocab.word2idx)).to(DEVICE)


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


# In[ ]:


#LSTM= 2layers+no bi
enc = EncoderRNN(feat_size=FEATURE_DIM, hidden_size=ENC_HIDDEN,num_layers=2, bidirectional=False).to(DEVICE)
dec = DecoderWithAttention(embed_size=EMBED_SIZE, enc_dim=enc.output_size, dec_hidden=DEC_HIDDEN, vocab_size=len(vocab.word2idx)).to(DEVICE)
#dec = DecoderWithBiLSTM(embed_size=EMBED_SIZE, enc_dim=enc.output_size, dec_hidden=DEC_HIDDEN, vocab_size=len(vocab.word2idx)).to(DEVICE)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1,ignore_index=vocab.word2idx[vocab.pad_token])
params = list(enc.parameters()) + list(dec.parameters())
optimizer = optim.Adam(params, lr=LR)

def train_one_epoch(train_loader, enc, dec, optimizer, criterion, device, clip=5.0):
    enc.train()
    dec.train()
    running_loss = 0.0

    for (global_feats, motion_feats, caps, cap_lens) in tqdm(train_loader):
        # Move tensors to device
        global_feats = global_feats.to(device)      # (B, 28, 2048)
        motion_feats = motion_feats.to(device)      # (B, 28, 64)
        caps = caps.to(device)                      # (B, max_caption_len)

        optimizer.zero_grad()

        # Pass both global and motion features to encoder
        encoder_outs = enc(global_feats, motion_feats)  # (B, 28, hidden_dim)

        # Decoder input: all tokens except last
        outputs = dec(encoder_outs, caps[:, :-1])       # (B, L-1, vocab_size)
        targets = caps[:, 1:]                           # (B, L-1)

        # Compute loss
        loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], clip)
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)



# In[ ]:


#LSTM= 2layers+Transformer without cross attention
enc = EncoderRNN(feat_size=FEATURE_DIM, hidden_size=ENC_HIDDEN, bidirectional=True).to(DEVICE)

dec = TransformerDecoderNoAttn(
    vocab_size=len(vocab.word2idx),
    embed_dim=512,
    enc_dim=enc.output_size, 
    num_layers=3 # 3,5,4
).to(DEVICE)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1,ignore_index=vocab.word2idx[vocab.pad_token])
params = list(enc.parameters()) + list(dec.parameters())
optimizer = optim.Adam(params, lr=LR)


def train_one_epoch(train_loader, encoder, decoder, optimizer, criterion, device, clip=5.0):
    encoder.train()
    decoder.train()

    running_loss = 0.0

    for feats, caps, cap_lens in tqdm(train_loader):
        feats = feats.to(device)          # (B, T, D)
        caps = caps.to(device)            # (B, L)

        optimizer.zero_grad()

        # ------------------------
        # 1. ENCODER FORWARD PASS
        # ------------------------
        enc_out, _ = encoder(feats)       # (B, T, enc_dim)

        # -----------------------------------------------------------
        # 2. TRANSFORMER DECODER — NO TEACHER FORCING
        # -----------------------------------------------------------
        outputs = decoder(enc_out, caps)  # (B, L, V)

        # -----------------------------------------------------------
        # 3. SHIFT OUTPUTS/TARGETS FOR LOSS
        # -----------------------------------------------------------
        outputs = outputs[:, :-1, :].contiguous() 
        targets = caps[:, 1:].contiguous()      

        loss = criterion(
            outputs.reshape(-1, outputs.size(-1)),
            targets.reshape(-1)
        )

        # Backprop
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)

        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)



# In[ ]:


# instantiate transformer+cross attention-3L
enc = EncoderRNN(feat_size=FEATURE_DIM, hidden_size=ENC_HIDDEN,num_layers=2, bidirectional=True).to(DEVICE)
dec = TransformerDecoderWithCrossAttn(
    vocab_size=len(vocab.word2idx),
    embed_dim=512,               
    enc_dim=enc.output_size,       
    num_layers=3,
    nhead=8,
    ff_dim=2048,
).to(DEVICE)




criterion = nn.CrossEntropyLoss(label_smoothing=0.1,ignore_index=vocab.word2idx[vocab.pad_token])
params = list(enc.parameters()) + list(dec.parameters())
optimizer = optim.Adam(params, lr=LR)


# In[8]:


pip install rouge


# In[10]:


pip install pycocoevalcap


# **Evaluate Function**

# In[30]:


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

    # -----------------------
    # Compute Metrics
    # -----------------------
    smoothie = SmoothingFunction().method4
    bleu4 = corpus_bleu(all_refs, all_preds, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

    refs_str = [' '.join(ref[0]) for ref in all_refs]
    preds_str = [' '.join(pred) for pred in all_preds]

    rouge_scores = rouge.get_scores(preds_str, refs_str, avg=True)
    cider_score, _ = cider_scorer.compute_score(
        {i: [refs_str[i]] for i in range(len(refs_str))},
        {i: [preds_str[i]] for i in range(len(preds_str))}
    )

    return bleu4, rouge_scores, cider_score, all_refs, all_preds


# **Training and Evaluation of the models for English Caption genearttion-3 features,BiLSTM Encoder+diffrent Decoders (1290 videos)**

# In[37]:


# -------------------------
 #Training Loop- BiLSTM ENC+LSTM DEC+LOUNG ATTENTION
# -------------------------
best_bleu = 0.0

for epoch in range(1, EPOCHS + 1):
    # ---- Training ----
    train_loss = train_one_epoch(train_loader, enc, dec, optimizer, criterion, DEVICE)
    print(f"Epoch {epoch} | Train Loss: {train_loss:.4f}")

    # ---- Validation ----
    bleu, rouge_scores, cider, _, _ = evaluate_with_metrics(val_loader, enc, dec, vocab, DEVICE, DEC_HIDDEN)
    rouge_l_f = rouge_scores['rouge-l']['f']  # Extract F1-score for ROUGE-L

    print(f"[Epoch {epoch}] Validation Metrics:")
    print(f"  BLEU-4 = {bleu:.4f}")
    print(f"  ROUGE-L (F1) = {rouge_l_f:.4f}")
    print(f"  CIDEr = {cider:.4f}")

    # ---- Save best model ----
    if bleu > best_bleu:
        best_bleu = bleu
        torch.save({
            'enc_state': enc.state_dict(),
            'dec_state': dec.state_dict(),
            'vocab': vocab.word2idx
        }, "best_checkpoint_1290_3f_eng.pth")
        print(" Saved new best checkpoint.")

print(f"\nTraining complete. Best BLEU-4 achieved: {best_bleu:.4f}")


# In[49]:


import torch
import numpy as np
import os
import random
from tqdm import tqdm


#Load the best saved checkpoint

ckpt = torch.load("/kaggle/working/best_checkpoint_1290_3f_eng.pth", map_location=DEVICE)
enc.load_state_dict(ckpt['enc_state'])
dec.load_state_dict(ckpt['dec_state'])
print(" Loaded checkpoint successfully.")

#Evaluate on test set

print("Running test evaluation (BLEU, ROUGE, CIDEr)...")
test_bleu, test_rouge, test_cider ,_,_= evaluate_with_metrics(test_loader, enc, dec, vocab, DEVICE)

test_rouge_l = test_rouge['rouge-l']['f']

print(f"\n📊 Test Metrics:")
print(f"BLEU-4  = {test_bleu:.4f}")
print(f"ROUGE-L = {test_rouge_l:.4f}")
print(f"CIDEr   = {test_cider:.4f}")



def generate_caption_for_video(video_feat, enc, dec, vocab, device, max_len=20):
    enc.eval()
    dec.eval()

    # Avoid warning about tensor creation
    if isinstance(video_feat, torch.Tensor):
        feat_tensor = video_feat.clone().detach().float().to(device)
    else:
        feat_tensor = torch.tensor(video_feat, dtype=torch.float32, device=device)

    # Ensure correct shape (B, T, D)
    if feat_tensor.dim() == 2:
        feat_tensor = feat_tensor.unsqueeze(0)   # (1, T, D)
    elif feat_tensor.dim() == 4:
        feat_tensor = feat_tensor.squeeze(0)     # (B, T, D)

    with torch.no_grad():
        encoder_outs, _ = enc(feat_tensor)

        # hidden, cell initialized same as training
        hidden = torch.zeros(1, 1, dec.lstm.hidden_size, device=device)
        cell = torch.zeros(1, 1, dec.lstm.hidden_size, device=device)

        input_word = torch.LongTensor([vocab.word2idx[vocab.bos_token]]).to(device)
        generated_tokens = []

        for _ in range(max_len):

            out, hidden, cell, attn_weights = dec.forward_step(
                input_word,
                hidden.squeeze(0),  # (B, hidden_size)
                cell.squeeze(0),    # (B, hidden_size)
                encoder_outs
            )

            # restore shape for next timestep
            hidden = hidden.unsqueeze(0)
            cell = cell.unsqueeze(0)

            next_word = out.argmax(1).item()
            if next_word == vocab.word2idx[vocab.eos_token]:
                break

            generated_tokens.append(vocab.idx2word.get(next_word, vocab.unk_token))
            input_word = torch.LongTensor([next_word]).to(device)

    return " ".join(generated_tokens)


#  Pick a few random test samples
import random


random.seed(9)

#  random 5 videos test_items
sample_videos = random.sample(list(test_items.keys()), min(5, len(test_items)))

for i, vid in enumerate(sample_videos):

    sample_feat = load_combined_features(vid, sample_frames=16)


    sample_feat_tensor = torch.FloatTensor(sample_feat).unsqueeze(0).to(DEVICE)  # (1, T, D)


    generated_caption = generate_caption_for_video(sample_feat_tensor, enc, dec, vocab, DEVICE)
    references = test_items[vid]  # list of ground-truth captions

    print(f"\n🎬 Video {i+1}: {vid}")
    print(f"📝 Generated Caption: {generated_caption}")
    print("📖 Reference Captions:")
    for j, ref in enumerate(references[:3]):
        print(f"  Ref {j+1}: {ref}")



# In[41]:


import os
import numpy as np
import torch

# Define feature folders
FEATURES_GLOBAL_DIR = "/kaggle/input/msr-vtt-1289-hindi-english/features_global/features_global"
FEATURES_LOCAL_DIR  = "/kaggle/input/msr-vtt-1289-hindi-english/features_local/features_local"
FEATURES_MOTION_DIR = "/kaggle/input/msr-vtt-1289-hindi-english/features_motion/features_motion"

def load_combined_features(vid, sample_frames=16):
    """
    Load and combine global, local, and motion features for a given video.
    Returns (T, 4608)
    """
    vid_id = vid.replace(".mp4", ".npy")

    global_path = os.path.join(FEATURES_GLOBAL_DIR, vid_id)
    local_path  = os.path.join(FEATURES_LOCAL_DIR, vid_id)
    motion_path = os.path.join(FEATURES_MOTION_DIR, vid_id)

    # Load features
    global_feat = np.load(global_path)    # (T, 2048)
    local_feat  = np.load(local_path)     # (T, 49, 2048)
    motion_feat = np.load(motion_path)    # (512,)

    # Mean-pool local features
    local_mean = local_feat.mean(axis=1)  # (T, 2048)

    # Repeat motion features per frame
    motion_repeat = np.repeat(motion_feat[np.newaxis, :], sample_frames, axis=0)  # (T, 512)

    # Concatenate all
    feats = np.concatenate([global_feat, local_mean, motion_repeat], axis=1)  # (T, 4608)

    # Pad/truncate frames
    if feats.shape[0] < sample_frames:
        pad = np.zeros((sample_frames - feats.shape[0], feats.shape[1]), dtype=np.float32)
        feats = np.concatenate([feats, pad], axis=0)
    else:
        feats = feats[:sample_frames]

    return feats.astype(np.float32)


# **Attention based Fusion of features**

# Code to load the dataset for attention based fusion

# In[9]:


#Code to load the dataset for attention based fusion
import os
import torch
import numpy as np
from torch.utils.data import Dataset

FEATURES_GLOBAL_DIR = "/kaggle/input/msr-vtt-1289-hindi-english/features_global/features_global"
FEATURES_LOCAL_DIR  = "/kaggle/input/msr-vtt-1289-hindi-english/features_local/features_local"
FEATURES_MOTION_DIR = "/kaggle/input/msr-vtt-1289-hindi-english/features_motion/features_motion"

class MSRVTTMultiFeatureDataset(Dataset):
    def __init__(self, captions_dict, vocab, sample_frames=16, max_caption_len=30):
        self.items = [(vid, c) for vid, caps in captions_dict.items() for c in caps]
        self.vocab = vocab
        self.sample_frames = sample_frames
        self.max_caption_len = max_caption_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        vid, cap = self.items[idx]
        vid_id = vid.replace(".mp4", ".npy")

        # Load features
        global_feat = np.load(os.path.join(FEATURES_GLOBAL_DIR, vid_id))  # (T,2048)
        local_feat  = np.load(os.path.join(FEATURES_LOCAL_DIR,  vid_id))  # (T,49,2048)
        motion_feat = np.load(os.path.join(FEATURES_MOTION_DIR, vid_id))  # (512,)

        # ---------- Fix shapes ----------
        if global_feat.ndim == 1:
            global_feat = global_feat[None, :]

        if local_feat.ndim == 3:
            local_mean = local_feat.mean(axis=1)              # (T,2048)
        elif local_feat.ndim == 2:
            local_mean = local_feat                            # (T,2048)
        elif local_feat.ndim == 1:
            local_mean = local_feat[None, :]                   # (1,2048)
        else:
            raise ValueError(f"Bad local_feat shape: {local_feat.shape}")

        if motion_feat.ndim == 1:
            motion_repeat = np.repeat(motion_feat[None, :], global_feat.shape[0], axis=0)  # (T,512)
        else:
            raise ValueError(f"Bad motion_feat shape: {motion_feat.shape}")

        # ---------- Pad or truncate all three ----------
        T = self.sample_frames

        def pad_to_T(x):
            if x.shape[0] < T:
                pad = np.zeros((T - x.shape[0], x.shape[1]), dtype=np.float32)
                return np.concatenate([x, pad], axis=0)
            else:
                return x[:T]

        global_feat = pad_to_T(global_feat).astype(np.float32)      # (T,2048)
        local_mean  = pad_to_T(local_mean).astype(np.float32)       # (T,2048)
        motion_repeat = pad_to_T(motion_repeat).astype(np.float32)  # (T,512)

        # ---------- Caption numericalization ----------
        numer = self.vocab.numericalize(cap)
        if len(numer) > self.max_caption_len:
            numer = numer[:self.max_caption_len-1] + \
                    [self.vocab.word2idx[self.vocab.eos_token]]

        cap_len = len(numer)
        pad_len = self.max_caption_len - cap_len
        if pad_len > 0:
            numer = numer + [self.vocab.word2idx[self.vocab.pad_token]] * pad_len

        return (
            torch.FloatTensor(global_feat),     # (T,2048)
            torch.FloatTensor(local_mean),      # (T,2048)
            torch.FloatTensor(motion_repeat),   # (T,512)
            torch.LongTensor(numer),            # (L)
            cap_len
        )

def collate_fn(batch):
    global_f = torch.stack([b[0] for b in batch], dim=0)
    local_f  = torch.stack([b[1] for b in batch], dim=0)
    motion_f = torch.stack([b[2] for b in batch], dim=0)
    caps     = torch.stack([b[3] for b in batch], dim=0)
    cap_lens = torch.LongTensor([b[4] for b in batch])
    return global_f, local_f, motion_f, caps, cap_lens

items = list(captions_filtered.items())
random.seed(42)
random.shuffle(items)
n = len(items)

train_items = dict(items[:int(0.8*n)])
val_items   = dict(items[int(0.8*n):int(0.9*n)])
test_items  = dict(items[int(0.9*n):])

train_ds = MSRVTTMultiFeatureDataset(train_items, vocab)
val_ds   = MSRVTTMultiFeatureDataset(val_items, vocab)
test_ds  = MSRVTTMultiFeatureDataset(test_items, vocab)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=collate_fn, num_workers=4)

val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                          collate_fn=collate_fn, num_workers=4)

test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                          collate_fn=collate_fn, num_workers=4)

print("Train size:", len(train_ds), "Val size:", len(val_ds), "Test size:", len(test_ds))


# In[18]:


#for hindi
import os
import torch
import numpy as np
from torch.utils.data import Dataset

FEATURES_GLOBAL_DIR = "/kaggle/input/msr-vtt-1289-hindi-english/features_global/features_global"
FEATURES_LOCAL_DIR  = "/kaggle/input/msr-vtt-1289-hindi-english/features_local/features_local"
FEATURES_MOTION_DIR = "/kaggle/input/msr-vtt-1289-hindi-english/features_motion/features_motion"

class MSRVTTMultiFeatureDataset(Dataset):
    def __init__(self, captions_dict, vocab, sample_frames=16, max_caption_len=30):
        self.items = [(vid, c) for vid, caps in captions_dict.items() for c in caps]
        self.vocab = vocab
        self.sample_frames = sample_frames
        self.max_caption_len = max_caption_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        vid, cap = self.items[idx]
        vid_id = vid.replace(".mp4", ".npy")

        # Load features
        global_feat = np.load(os.path.join(FEATURES_GLOBAL_DIR, vid_id))  # (T,2048)
        local_feat  = np.load(os.path.join(FEATURES_LOCAL_DIR,  vid_id))  # (T,49,2048)
        motion_feat = np.load(os.path.join(FEATURES_MOTION_DIR, vid_id))  # (512,)

        # ---------- Fix shapes ----------
        if global_feat.ndim == 1:
            global_feat = global_feat[None, :]

        if local_feat.ndim == 3:
            local_mean = local_feat.mean(axis=1)              # (T,2048)
        elif local_feat.ndim == 2:
            local_mean = local_feat                            # (T,2048)
        elif local_feat.ndim == 1:
            local_mean = local_feat[None, :]                   # (1,2048)
        else:
            raise ValueError(f"Bad local_feat shape: {local_feat.shape}")

        if motion_feat.ndim == 1:
            motion_repeat = np.repeat(motion_feat[None, :], global_feat.shape[0], axis=0)  # (T,512)
        else:
            raise ValueError(f"Bad motion_feat shape: {motion_feat.shape}")

        # ---------- Pad or truncate all three ----------
        T = self.sample_frames

        def pad_to_T(x):
            if x.shape[0] < T:
                pad = np.zeros((T - x.shape[0], x.shape[1]), dtype=np.float32)
                return np.concatenate([x, pad], axis=0)
            else:
                return x[:T]

        global_feat = pad_to_T(global_feat).astype(np.float32)      # (T,2048)
        local_mean  = pad_to_T(local_mean).astype(np.float32)       # (T,2048)
        motion_repeat = pad_to_T(motion_repeat).astype(np.float32)  # (T,512)

        # ---------- Caption numericalization ----------
        numer = self.vocab.numericalize(cap)
        if len(numer) > self.max_caption_len:
            numer = numer[:self.max_caption_len-1] + \
                    [self.vocab.word2idx[self.vocab.eos_token]]

        cap_len = len(numer)
        pad_len = self.max_caption_len - cap_len
        if pad_len > 0:
            numer = numer + [self.vocab.word2idx[self.vocab.pad_token]] * pad_len

        return (
            torch.FloatTensor(global_feat),     # (T,2048)
            torch.FloatTensor(local_mean),      # (T,2048)
            torch.FloatTensor(motion_repeat),   # (T,512)
            torch.LongTensor(numer),            # (L)
            cap_len
        )

def collate_fn(batch):
    global_f = torch.stack([b[0] for b in batch], dim=0)
    local_f  = torch.stack([b[1] for b in batch], dim=0)
    motion_f = torch.stack([b[2] for b in batch], dim=0)
    caps     = torch.stack([b[3] for b in batch], dim=0)
    cap_lens = torch.LongTensor([b[4] for b in batch])
    return global_f, local_f, motion_f, caps, cap_lens

items = list(captions_hindi.items())
random.seed(42)
random.shuffle(items)
n = len(items)

train_items = dict(items[:int(0.8*n)])
val_items   = dict(items[int(0.8*n):int(0.9*n)])
test_items  = dict(items[int(0.9*n):])

train_ds = MSRVTTMultiFeatureDataset(train_items, vocab)
val_ds   = MSRVTTMultiFeatureDataset(val_items, vocab)
test_ds  = MSRVTTMultiFeatureDataset(test_items, vocab)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=collate_fn, num_workers=4)

val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                          collate_fn=collate_fn, num_workers=4)

test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                          collate_fn=collate_fn, num_workers=4)

print("Train size:", len(train_ds), "Val size:", len(val_ds), "Test size:", len(test_ds))


# In[87]:


#Additive (Bahdanau) attention for multimodal fusion
class FusionAttention(nn.Module):
    def __init__(self, dim_g, dim_l, dim_m, fused_dim):
        super().__init__()
        self.Wg = nn.Linear(dim_g, fused_dim)
        self.Wl = nn.Linear(dim_l, fused_dim)
        self.Wm = nn.Linear(dim_m, fused_dim)
        self.v  = nn.Linear(fused_dim, 1)

    def forward(self, g, l, m):
        # g, l, m = (B, T, D?)

        # Project inputs to fused_dim
        g_proj = self.Wg(g)  # (B, T, fused_dim)
        l_proj = self.Wl(l)  # (B, T, fused_dim)
        m_proj = self.Wm(m)  # (B, T, fused_dim)

        # Compute attention scores on projected inputs
        score_g = self.v(torch.tanh(g_proj))  # (B, T, 1)
        score_l = self.v(torch.tanh(l_proj))  # (B, T, 1)
        score_m = self.v(torch.tanh(m_proj))  # (B, T, 1)

        # Calculate attention weights
        αg = torch.softmax(score_g, dim=1)  # (B, T, 1)
        αl = torch.softmax(score_l, dim=1)
        αm = torch.softmax(score_m, dim=1)

        # Fuse features with attention weights applied to projected features
        fused = αg * g_proj + αl * l_proj + αm * m_proj  # (B, T, fused_dim)

        return fused


# In[4]:


#Parameters used for Transformer based models+ multimodal attention fusion
BATCH_SIZE = 32
GLOBAL_DIM = 2048
LOCAL_DIM = 2048
MOTION_DIM = 512
FEATURE_DIM = 4608    # global + local + motion

FUSION_HIDDEN = 512
NUM_HEADS = 8

ENC_HIDDEN = 512
ENC_LAYERS = 1
ENC_BIDIRECTIONAL = True

EMBED_SIZE = 300
DEC_HIDDEN = 512

LR = 1e-4
EPOCHS = 15
CLIP = 5.0
TEACHER_FORCING = 0.75
LABEL_SMOOTHING = 0.1

SAMPLE_FRAMES = 16
MAX_CAPTION_LEN = 30
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[90]:


#initiation of encoder,decoder and fusion module
fusion_module = FusionAttention(
    dim_g=2048,     
    dim_l=2048,     
    dim_m=512,   
    fused_dim=512   
).to(DEVICE)


enc = EncoderRNN(
    feat_size=512, 
    hidden_size=ENC_HIDDEN, 
    bidirectional=True
).to(DEVICE)

dec = DecoderWithAttention(
    embed_size=EMBED_SIZE,
    enc_dim=enc.output_size,
    dec_hidden=DEC_HIDDEN,
    vocab_size=len(vocab.word2idx)
).to(DEVICE)

def train_one_epoch(train_loader, fusion_module, enc, dec, optimizer, criterion, device, clip=5.0):
    enc.train(); dec.train(); fusion_module.train()
    running_loss = 0.0

    for global_f, local_f, motion_f, caps, cap_lens in tqdm(train_loader):
        global_f = global_f.to(device)   # (B, T, 2048)
        local_f  = local_f.to(device)    # (B, T, 2048)
        motion_f = motion_f.to(device)   # (B, T, 512)
        caps     = caps.to(device)       # (B, L)

        optimizer.zero_grad()


        fused_feats = fusion_module(global_f, local_f, motion_f)   # (B, T, F)

        # Encoder
        encoder_outs, _ = enc(fused_feats)

        # Decoder
        outputs, _ = dec(encoder_outs, caps, teacher_forcing_ratio=0.75)


        outputs = outputs[:, 1:, :].contiguous()
        targets = caps[:, 1:].contiguous()

        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + 
                                       list(dec.parameters()) + 
                                       list(fusion_module.parameters()), clip)

        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(train_loader)

params = list(fusion_module.parameters()) + list(enc.parameters()) + list(dec.parameters())
optimizer = optim.Adam(params, lr=LR)
criterion = nn.CrossEntropyLoss(
    ignore_index=vocab.word2idx[vocab.pad_token],label_smoothing=LABEL_SMOOTHING)


# In[91]:


import torch
from tqdm import tqdm
import random
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge import Rouge
from pycocoevalcap.cider.cider import Cider
def evaluate_with_metrics(loader, fusion_module, enc, dec, vocab, device, DEC_HIDDEN=512):
    enc.eval()
    dec.eval()
    fusion_module.eval()

    all_refs, all_preds = [], []

    rouge = Rouge()
    cider_scorer = Cider()

    with torch.no_grad():
        for global_f, local_f, motion_f, caps, cap_lens in tqdm(loader, desc="Evaluating"):

            global_f = global_f.to(device)    # (B, T, 2048)
            local_f  = local_f.to(device)     # (B, T, 2048)
            motion_f = motion_f.to(device)    # (B, T, 512)
            caps     = caps.to(device)        # (B, L)

            B = global_f.size(0)

           # Apply multimodal fusion

            fused_feats = fusion_module(global_f, local_f, motion_f)  # (B, T, F)


            # Encoder Forward
            encoder_outs, _ = enc(fused_feats)

            # Initialize hidden & cell
            hidden = torch.zeros(B, DEC_HIDDEN, device=device)
            cell   = torch.zeros(B, DEC_HIDDEN, device=device)

            # Start token <BOS>
            input_word = torch.LongTensor([vocab.word2idx[vocab.bos_token]] * B).to(device)

            preds = [[] for _ in range(B)]
            max_len = caps.size(1)


            # Greedy Decoding
            for t in range(1, max_len):
                out, hidden, cell, attn_weights = dec.forward_step(
                    input_word, hidden, cell, encoder_outs
                )
                top1 = out.argmax(1)
                input_word = top1

                for i in range(B):
                    preds[i].append(top1[i].item())


            # Convert Predictions to Words

            for i in range(B):
                pred_tokens = []
                for tok in preds[i]:
                    if tok in (vocab.word2idx[vocab.pad_token], vocab.word2idx[vocab.bos_token]):
                        continue
                    if tok == vocab.word2idx[vocab.eos_token]:
                        break
                    pred_tokens.append(vocab.idx2word.get(tok, vocab.unk_token))
                all_preds.append(pred_tokens)

                # Reference caption
                ref_tokens = []
                for tok in caps[i].cpu().numpy():
                    if tok in (vocab.word2idx[vocab.pad_token], vocab.word2idx[vocab.bos_token]):
                        continue
                    if tok == vocab.word2idx[vocab.eos_token]:
                        break
                    ref_tokens.append(vocab.idx2word.get(int(tok), vocab.unk_token))
                all_refs.append([ref_tokens])


    smoothie = SmoothingFunction().method4
    bleu4 = corpus_bleu(all_refs, all_preds, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

    refs_str = [' '.join(ref[0]) for ref in all_refs]
    preds_str = [' '.join(pred) for pred in all_preds]

    rouge_scores = rouge.get_scores(preds_str, refs_str, avg=True)

    cider_score, _ = cider_scorer.compute_score(
        {i: [refs_str[i]] for i in range(len(refs_str))},
        {i: [preds_str[i]] for i in range(len(preds_str))}
    )

    return bleu4, rouge_scores, cider_score, all_refs, all_preds


# In[16]:


# Training Loop

best_bleu = 0.0

for epoch in range(1, EPOCHS + 1):

    # ---- Training ----
    train_loss = train_one_epoch(train_loader, fusion_module, enc, dec, optimizer, criterion, DEVICE)
    print(f"Epoch {epoch} | Train Loss: {train_loss:.4f}")

    # ---- Validation ----
    bleu, rouge_scores, cider, _, _ = evaluate_with_metrics(
        val_loader,
        fusion_module,
        enc,
        dec,
        vocab,
        DEVICE,
        DEC_HIDDEN
    )

    rouge_l_f = rouge_scores['rouge-l']['f']

    print(f"[Epoch {epoch}] Validation Metrics:")
    print(f"  BLEU-4 = {bleu:.4f}")
    print(f"  ROUGE-L (F1) = {rouge_l_f:.4f}")
    print(f"  CIDEr = {cider:.4f}")

    #Save best model 
    if bleu > best_bleu:
        best_bleu = bleu
        torch.save({
            'fusion_state': fusion_module.state_dict(),
            'enc_state': enc.state_dict(),
            'dec_state': dec.state_dict(),
            'vocab': vocab.word2idx
        }, "best_checkpoint_multifeature_attentionfusion.pth")
        print(" Saved new best checkpoint.")

print(f"\nTraining complete. Best BLEU-4 achieved: {best_bleu:.4f}")


# In[19]:


def generate_caption_for_video(global_f, local_f, motion_f, fusion_module, enc, dec, vocab, device, max_len=20):
    enc.eval()
    dec.eval()
    fusion_module.eval()

    # Move inputs to device
    global_f = global_f.to(device)
    local_f = local_f.to(device)
    motion_f = motion_f.to(device)

    with torch.no_grad():

        fused_feats = fusion_module(global_f, local_f, motion_f)  # (1, T, fused_dim)

        # Encoder forward
        encoder_outs, _ = enc(fused_feats)

        # Initialize hidden and cell states as during training
        hidden = torch.zeros(1, 1, dec.lstm.hidden_size, device=device)
        cell = torch.zeros(1, 1, dec.lstm.hidden_size, device=device)

        input_word = torch.LongTensor([vocab.word2idx[vocab.bos_token]]).to(device)
        generated_tokens = []

        for _ in range(max_len):
            out, hidden, cell, attn_weights = dec.forward_step(
                input_word, hidden.squeeze(0), cell.squeeze(0), encoder_outs
            )
            hidden = hidden.unsqueeze(0)
            cell = cell.unsqueeze(0)

            next_word = out.argmax(1).item()
            if next_word == vocab.word2idx[vocab.eos_token]:
                break

            generated_tokens.append(vocab.idx2word.get(next_word, vocab.unk_token))
            input_word = torch.LongTensor([next_word]).to(device)

    return " ".join(generated_tokens)


# In[18]:


import torch
from tqdm import tqdm


# Load checkpoint with fusion, enc, dec states

checkpoint_path = "/kaggle/working/best_checkpoint_multifeature_attentionfusion.pth"
ckpt = torch.load(checkpoint_path, map_location=DEVICE)

fusion_module.load_state_dict(ckpt['fusion_state'])
enc.load_state_dict(ckpt['enc_state'])
dec.load_state_dict(ckpt['dec_state'])
vocab.word2idx = ckpt.get('vocab', vocab.word2idx)

print("Loaded fusion, encoder, decoder states from checkpoint.")

# Set models to evaluation mode
fusion_module.eval()
enc.eval()
dec.eval()



print("Running test evaluation (BLEU, ROUGE, CIDEr)...")

test_bleu, test_rouge, test_cider, _, _ = evaluate_with_metrics(
    test_loader, fusion_module, enc, dec, vocab, DEVICE, DEC_HIDDEN
)

print(f"\n📊 Test Metrics:")
print(f"BLEU-4  = {test_bleu:.4f}")
print(f"ROUGE-L = {test_rouge['rouge-l']['f']:.4f}")
print(f"CIDEr   = {test_cider:.4f}")


# In[32]:


import torch
import numpy as np
import os
import random
from tqdm import tqdm


# Load the checkpoint
checkpoint_path = "/kaggle/working/best_checkpoint_multifeature_attentionfusion.pth"
ckpt = torch.load(checkpoint_path, map_location=DEVICE)

fusion_module.load_state_dict(ckpt['fusion_state'])
enc.load_state_dict(ckpt['enc_state'])
dec.load_state_dict(ckpt['dec_state'])

print(" Loaded fusion, encoder, decoder from checkpoint successfully.")

# Set to evaluation mode
fusion_module.eval()
enc.eval()
dec.eval()


def load_global_feature(vid, sample_frames=16):

    vid_id = vid.replace(".mp4", ".npy")
    global_feat = np.load(os.path.join(FEATURES_GLOBAL_DIR, vid_id))  # (T, 2048)

    if global_feat.ndim == 1:
        global_feat = global_feat[None, :]

    # Pad or truncate to sample_frames
    if global_feat.shape[0] < sample_frames:
        pad = np.zeros((sample_frames - global_feat.shape[0], global_feat.shape[1]), dtype=np.float32)
        global_feat = np.concatenate([global_feat, pad], axis=0)
    else:
        global_feat = global_feat[:sample_frames]

    return global_feat.astype(np.float32)

def load_local_feature(vid, sample_frames=16):

    vid_id = vid.replace(".mp4", ".npy")
    local_feat = np.load(os.path.join(FEATURES_LOCAL_DIR, vid_id))  # (T, 49, 2048) or (T, 2048)

    # Handle different shapes
    if local_feat.ndim == 3:
        local_feat = local_feat.mean(axis=1)  # (T, 2048)
    elif local_feat.ndim == 1:
        local_feat = local_feat[None, :]

    # Pad or truncate to sample_frames
    if local_feat.shape[0] < sample_frames:
        pad = np.zeros((sample_frames - local_feat.shape[0], local_feat.shape[1]), dtype=np.float32)
        local_feat = np.concatenate([local_feat, pad], axis=0)
    else:
        local_feat = local_feat[:sample_frames]

    return local_feat.astype(np.float32)

def load_motion_feature(vid, sample_frames=16):

    vid_id = vid.replace(".mp4", ".npy")
    motion_feat = np.load(os.path.join(FEATURES_MOTION_DIR, vid_id))  # (512,)

    if motion_feat.ndim == 1:
        # Repeat motion feature across time dimension
        motion_feat = np.repeat(motion_feat[None, :], sample_frames, axis=0)  # (T, 512)

    # Pad or truncate to sample_frames
    if motion_feat.shape[0] < sample_frames:
        pad = np.zeros((sample_frames - motion_feat.shape[0], motion_feat.shape[1]), dtype=np.float32)
        motion_feat = np.concatenate([motion_feat, pad], axis=0)
    else:
        motion_feat = motion_feat[:sample_frames]

    return motion_feat.astype(np.float32)


# Caption generation function with fusion
def generate_caption_for_video(global_f, local_f, motion_f, fusion_module, enc, dec, vocab, device, max_len=20):

    fusion_module.eval()
    enc.eval()
    dec.eval()

    # Move to device
    global_f = global_f.to(device)
    local_f = local_f.to(device)
    motion_f = motion_f.to(device)

    with torch.no_grad():
        # Fuse features
        fused_feats = fusion_module(global_f, local_f, motion_f)  # (1, T, 512)

        #  Encode fused features
        encoder_outs, _ = enc(fused_feats)  # (1, T, enc_output_dim)

        #  Initialize decoder hidden and cell states
        B = 1
        hidden = torch.zeros(B, dec.lstm.hidden_size, device=device)
        cell = torch.zeros(B, dec.lstm.hidden_size, device=device)

        # Start with <BOS> token
        input_word = torch.LongTensor([vocab.word2idx[vocab.bos_token]]).to(device)
        generated_tokens = []

        #  Generate caption token by token
        for _ in range(max_len):
            out, hidden, cell, attn_weights = dec.forward_step(
                input_word, hidden, cell, encoder_outs
            )

            next_word = out.argmax(1).item()

            # Stop if <EOS> token generated
            if next_word == vocab.word2idx[vocab.eos_token]:
                break

            generated_tokens.append(vocab.idx2word.get(next_word, vocab.unk_token))
            input_word = torch.LongTensor([next_word]).to(device)

    return " ".join(generated_tokens)


# Generate captions for sample test videos
random.seed(9)
sample_videos = random.sample(list(test_items.keys()), min(5, len(test_items)))

print("\n" + "="*60)
print(" Generating Captions for Sample Test Videos")
print("="*60)

for i, vid in enumerate(sample_videos):

    global_f = load_global_feature(vid, sample_frames=16)   # (16, 2048)
    local_f = load_local_feature(vid, sample_frames=16)     # (16, 2048)
    motion_f = load_motion_feature(vid, sample_frames=16)   # (16, 512)

    # Convert to tensors with batch dimension
    global_f = torch.FloatTensor(global_f).unsqueeze(0)  # (1, 16, 2048)
    local_f = torch.FloatTensor(local_f).unsqueeze(0)    # (1, 16, 2048)
    motion_f = torch.FloatTensor(motion_f).unsqueeze(0)  # (1, 16, 512)

    # Generate caption using fusion module
    generated_caption = generate_caption_for_video(
        global_f, local_f, motion_f, fusion_module, enc, dec, vocab, DEVICE
    )


    references = test_items[vid]

    print(f"\n Video {i+1}: {vid}")
    print(f" Generated Caption: {generated_caption}")
    print(" Reference Captions:")
    for j, ref in enumerate(references[:3]):
        print(f"   Ref {j+1}: {ref}")

print("\n" + "="*60)


# In[37]:


import torch
import numpy as np
import os

# ==========================================
# UNIFIED FEATURE LOADING & PREPROCESSING
# ==========================================
def load_and_preprocess_features(vid, sample_frames=16, 
                                  global_dir=FEATURES_GLOBAL_DIR,
                                  local_dir=FEATURES_LOCAL_DIR,
                                  motion_dir=FEATURES_MOTION_DIR):

    vid_id = vid.replace(".mp4", ".npy")

    # Load global features
    global_feat = np.load(os.path.join(global_dir, vid_id))  # (T, 2048)
    if global_feat.ndim == 1:
        global_feat = global_feat[None, :]

    # Load local features
    local_feat = np.load(os.path.join(local_dir, vid_id))  # (T, 49, 2048) or (T, 2048)
    if local_feat.ndim == 3:
        local_feat = local_feat.mean(axis=1)  # (T, 2048)
    elif local_feat.ndim == 1:
        local_feat = local_feat[None, :]

    # Load motion features
    motion_feat = np.load(os.path.join(motion_dir, vid_id))  # (512,)
    if motion_feat.ndim == 1:
        motion_feat = np.repeat(motion_feat[None, :], max(global_feat.shape[0], local_feat.shape[0]), axis=0)

    # Ensure all features have same temporal length
    T_max = max(global_feat.shape[0], local_feat.shape[0], motion_feat.shape[0])

    # Pad or truncate all to sample_frames
    def pad_to_T(x, T):
        if x.shape[0] < T:
            pad = np.zeros((T - x.shape[0], x.shape[1]), dtype=np.float32)
            return np.concatenate([x, pad], axis=0)
        else:
            return x[:T].astype(np.float32)

    global_feat = pad_to_T(global_feat, sample_frames)  # (sample_frames, 2048)
    local_feat = pad_to_T(local_feat, sample_frames)    # (sample_frames, 2048)
    motion_feat = pad_to_T(motion_feat, sample_frames)  # (sample_frames, 512)

    return global_feat, local_feat, motion_feat


# UNIFIED CAPTION GENERATION

def generate_caption_for_video(vid, fusion_module, enc, dec, vocab, device, 
                               sample_frames=16, max_len=20):

    fusion_module.eval()
    enc.eval()
    dec.eval()

    # Load and preprocess features using unified function
    global_f, local_f, motion_f = load_and_preprocess_features(vid, sample_frames=sample_frames)

    # Convert to tensors with batch dimension
    global_f = torch.FloatTensor(global_f).unsqueeze(0).to(device)  # (1, T, 2048)
    local_f = torch.FloatTensor(local_f).unsqueeze(0).to(device)    # (1, T, 2048)
    motion_f = torch.FloatTensor(motion_f).unsqueeze(0).to(device)  # (1, T, 512)

    with torch.no_grad():
        # Fuse features
        fused_feats = fusion_module(global_f, local_f, motion_f)  # (1, T, 512)

        # Encode
        encoder_outs, _ = enc(fused_feats)

        # Initialize decoder
        B = 1
        hidden = torch.zeros(B, dec.lstm.hidden_size, device=device)
        cell = torch.zeros(B, dec.lstm.hidden_size, device=device)

        input_word = torch.LongTensor([vocab.word2idx[vocab.bos_token]]).to(device)
        generated_tokens = []

        # Decode
        for _ in range(max_len):
            out, hidden, cell, _ = dec.forward_step(input_word, hidden, cell, encoder_outs)
            next_word = out.argmax(1).item()

            if next_word == vocab.word2idx[vocab.eos_token]:
                break

            generated_tokens.append(vocab.idx2word.get(next_word, vocab.unk_token))
            input_word = torch.LongTensor([next_word]).to(device)

    return " ".join(generated_tokens)


# SAMPLE GENERATION

random.seed(60)
sample_videos = random.sample(list(test_items.keys()), min(5, len(test_items)))

print("\n" + "="*60)
print(" Generating Captions for Sample Test Videos")
print("="*60)

for i, vid in enumerate(sample_videos):
    generated_caption = generate_caption_for_video(
        vid, fusion_module, enc, dec, vocab, DEVICE, sample_frames=16, max_len=20
    )
    references = test_items[vid]

    print(f"\n Video {i+1}: {vid}")
    print(f" Generated Caption: {generated_caption}")
    print(" Reference Captions:")
    for j, ref in enumerate(references[:3]):
        print(f"   Ref {j+1}: {ref}")

print("\n" + "="*60)


# In[47]:


random.seed(56)
sample_videos = random.sample(list(test_items.keys()), min(5, len(test_items)))

print("\n" + "="*60)
print(" Generating Captions for Sample Test Videos")
print("="*60)

for i, vid in enumerate(sample_videos):
    generated_caption = generate_caption_for_video(
        vid, fusion_module, enc, dec, vocab, DEVICE, max_len=20
    )
    references = test_items[vid]

    print(f"\n Video {i+1}: {vid}")
    print(f" Generated Caption: {generated_caption}")
    print(" Reference Captions:")
    for j, ref in enumerate(references[:3]):
        print(f"   Ref {j+1}: {ref}")

print("\n" + "="*60)


# In[92]:


#Multimodal attention based fusion-hindi
# -------------------------
# Training Loop
# -------------------------
best_bleu = 0.0

for epoch in range(1, EPOCHS + 1):

    # ---- Training ----
    train_loss = train_one_epoch(train_loader, fusion_module, enc, dec, optimizer, criterion, DEVICE)
    print(f"Epoch {epoch} | Train Loss: {train_loss:.4f}")

    # ---- Validation ----
    bleu, rouge_scores, cider, _, _ = evaluate_with_metrics(
        val_loader,
        fusion_module,
        enc,
        dec,
        vocab,
        DEVICE,
        DEC_HIDDEN
    )

    rouge_l_f = rouge_scores['rouge-l']['f']

    print(f"[Epoch {epoch}] Validation Metrics:")
    print(f"  BLEU-4 = {bleu:.4f}")
    print(f"  ROUGE-L (F1) = {rouge_l_f:.4f}")
    print(f"  CIDEr = {cider:.4f}")

    # ---- Save best model ----
    if bleu > best_bleu:
        best_bleu = bleu
        torch.save({
            'fusion_state': fusion_module.state_dict(),
            'enc_state': enc.state_dict(),
            'dec_state': dec.state_dict(),
            'vocab': vocab.word2idx
        }, "best_checkpoint_multifeature_attention_fusion_hindi.pth")
        print(" Saved new best checkpoint.")

print(f"\nTraining complete. Best BLEU-4 achieved: {best_bleu:.4f}")


# In[93]:


import torch
from tqdm import tqdm


# Load checkpoint with fusion, enc, dec states

checkpoint_path = "/kaggle/working/best_checkpoint_multifeature_attention_fusion_hindi.pth"
ckpt = torch.load(checkpoint_path, map_location=DEVICE)

fusion_module.load_state_dict(ckpt['fusion_state'])
enc.load_state_dict(ckpt['enc_state'])
dec.load_state_dict(ckpt['dec_state'])
vocab.word2idx = ckpt.get('vocab', vocab.word2idx)

print("Loaded fusion, encoder, decoder states from checkpoint.")


fusion_module.eval()
enc.eval()
dec.eval()



# Run evaluation on test set

print("Running test evaluation (BLEU, ROUGE, CIDEr)...")

test_bleu, test_rouge, test_cider, _, _ = evaluate_with_metrics(
    test_loader, fusion_module, enc, dec, vocab, DEVICE, DEC_HIDDEN
)

print(f"\n📊 Test Metrics:")
print(f"BLEU-4  = {test_bleu:.4f}")
print(f"ROUGE-L = {test_rouge['rouge-l']['f']:.4f}")
print(f"CIDEr   = {test_cider:.4f}")


# In[96]:


import torch
import numpy as np
import os
import random
from tqdm import tqdm

def clean_caption(spm_text):

    # Remove SentencePiece underscore markers
    text = spm_text.replace("▁", " ").strip()

    # Replace <unk> tokens with placeholder or remove
    text = text.replace("<unk>", "")  # Or use "[UNK]" or any custom token

    # Remove multiple spaces
    text = ' '.join(text.split())

    return text

# Load the checkpoint
checkpoint_path = "/kaggle/working/best_checkpoint_multifeature_attention_fusion_hindi.pth"
ckpt = torch.load(checkpoint_path, map_location=DEVICE)

fusion_module.load_state_dict(ckpt['fusion_state'])
enc.load_state_dict(ckpt['enc_state'])
dec.load_state_dict(ckpt['dec_state'])

print(" Loaded fusion, encoder, decoder from checkpoint successfully.")


fusion_module.eval()
enc.eval()
dec.eval()


def load_global_feature(vid, sample_frames=16):

    vid_id = vid.replace(".mp4", ".npy")
    global_feat = np.load(os.path.join(FEATURES_GLOBAL_DIR, vid_id))  # (T, 2048)

    if global_feat.ndim == 1:
        global_feat = global_feat[None, :]

    # Pad or truncate to sample_frames
    if global_feat.shape[0] < sample_frames:
        pad = np.zeros((sample_frames - global_feat.shape[0], global_feat.shape[1]), dtype=np.float32)
        global_feat = np.concatenate([global_feat, pad], axis=0)
    else:
        global_feat = global_feat[:sample_frames]

    return global_feat.astype(np.float32)

def load_local_feature(vid, sample_frames=16):

    vid_id = vid.replace(".mp4", ".npy")
    local_feat = np.load(os.path.join(FEATURES_LOCAL_DIR, vid_id))  # (T, 49, 2048) or (T, 2048)

    # Handle different shapes
    if local_feat.ndim == 3:
        local_feat = local_feat.mean(axis=1)  # (T, 2048)
    elif local_feat.ndim == 1:
        local_feat = local_feat[None, :]

    # Pad or truncate to sample_frames
    if local_feat.shape[0] < sample_frames:
        pad = np.zeros((sample_frames - local_feat.shape[0], local_feat.shape[1]), dtype=np.float32)
        local_feat = np.concatenate([local_feat, pad], axis=0)
    else:
        local_feat = local_feat[:sample_frames]

    return local_feat.astype(np.float32)

def load_motion_feature(vid, sample_frames=16):

    vid_id = vid.replace(".mp4", ".npy")
    motion_feat = np.load(os.path.join(FEATURES_MOTION_DIR, vid_id))  # (512,)

    if motion_feat.ndim == 1:
        # Repeat motion feature across time dimension
        motion_feat = np.repeat(motion_feat[None, :], sample_frames, axis=0)  # (T, 512)

    # Pad or truncate to sample_frames
    if motion_feat.shape[0] < sample_frames:
        pad = np.zeros((sample_frames - motion_feat.shape[0], motion_feat.shape[1]), dtype=np.float32)
        motion_feat = np.concatenate([motion_feat, pad], axis=0)
    else:
        motion_feat = motion_feat[:sample_frames]

    return motion_feat.astype(np.float32)


def generate_caption_for_video(global_f, local_f, motion_f, fusion_module, enc, dec, vocab, device, max_len=20):

    fusion_module.eval()
    enc.eval()
    dec.eval()

    # Move to device
    global_f = global_f.to(device)
    local_f = local_f.to(device)
    motion_f = motion_f.to(device)

    with torch.no_grad():
        # Step 1: Fuse features
        fused_feats = fusion_module(global_f, local_f, motion_f)  # (1, T, 512)

        # Step 2: Encode fused features
        encoder_outs, _ = enc(fused_feats)  # (1, T, enc_output_dim)

        # Step 3: Initialize decoder hidden and cell states
        B = 1
        hidden = torch.zeros(B, dec.lstm.hidden_size, device=device)
        cell = torch.zeros(B, dec.lstm.hidden_size, device=device)

        # Start with <BOS> token
        input_word = torch.LongTensor([vocab.word2idx[vocab.bos_token]]).to(device)
        generated_tokens = []

        # Step 4: Generate caption token by token
        for _ in range(max_len):
            out, hidden, cell, attn_weights = dec.forward_step(
                input_word, hidden, cell, encoder_outs
            )

            next_word = out.argmax(1).item()

            # Stop if <EOS> token generated
            if next_word == vocab.word2idx[vocab.eos_token]:
                break

            generated_tokens.append(vocab.idx2word.get(next_word, vocab.unk_token))
            input_word = torch.LongTensor([next_word]).to(device)

    return " ".join(generated_tokens)

# Generate captions for sample test videos

random.seed(70)
sample_videos = random.sample(list(test_items.keys()), min(5, len(test_items)))

print("\n" + "="*60)
print(" Generating Captions for Sample Test Videos")
print("="*60)

for i, vid in enumerate(sample_videos):
    # Load individual features for this video
    global_f = load_global_feature(vid, sample_frames=16)   # (16, 2048)
    local_f = load_local_feature(vid, sample_frames=16)     # (16, 2048)
    motion_f = load_motion_feature(vid, sample_frames=16)   # (16, 512)

    # Convert to tensors with batch dimension
    global_f = torch.FloatTensor(global_f).unsqueeze(0)  # (1, 16, 2048)
    local_f = torch.FloatTensor(local_f).unsqueeze(0)    # (1, 16, 2048)
    motion_f = torch.FloatTensor(motion_f).unsqueeze(0)  # (1, 16, 512)

    # Generate caption using fusion module
    generated_caption = generate_caption_for_video(
        global_f, local_f, motion_f, fusion_module, enc, dec, vocab, DEVICE
    )

    # Get reference captions
    references = test_items[vid]

    print(f"\n Video {i+1}: {vid}")
    print(f" Generated Caption: {clean_caption(generated_caption)}")
    print(" Reference Captions:")
    for j, ref in enumerate(references[:3]):
        print(f"   Ref {j+1}: {ref}")

print("\n" + "="*60)


# **Experiments using different Models for HINDI CAPTION GENERATION**

# In[16]:


pip install googletrans==4.0.0rc1


# In[21]:


import json
from googletrans import Translator
from tqdm import tqdm
import asyncio
import nest_asyncio

nest_asyncio.apply()

async def translate_caption(translator, cap):
    translation = await translator.translate(cap, src='en', dest='hi')
    return translation.text

async def translate_all_captions():
    with open('/kaggle/input/msr-vtt-1289-hindi-english/captions.json', 'r', encoding='utf-8') as f:
        captions_en = json.load(f)

    translator = Translator()
    captions_hi = {}

    video_keys = list(captions_en.keys())
    print(f"Number of video keys in JSON: {len(video_keys)}")

    for vid_key in tqdm(video_keys):
        en_caps = captions_en[vid_key]
        hi_caps = []
        for cap in en_caps:
            text = await translate_caption(translator, cap)
            hi_caps.append(text)

        captions_hi[vid_key] = hi_caps

    with open('captions_hindi.json', 'w', encoding='utf-8') as f:
        json.dump(captions_hi, f, ensure_ascii=False, indent=4)
    print("Finished Hindi caption translation and saved.")

# Run the asyncio function with await (only inside Jupyter or IPython)
await translate_all_captions()


# In[24]:


import json
from transformers import MarianMTModel, MarianTokenizer
import torch
from tqdm import tqdm

# Load tokenizer and model for English->Hindi translation
model_name = 'Helsinki-NLP/opus-mt-en-hi'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Load English captions JSON
with open('/kaggle/input/msr-vtt-1289-hindi-english/captions.json', 'r', encoding='utf-8') as f:
    captions_en = json.load(f)

captions_hi = {}

num_videos = 1290  # total number of videos expected

for vid_idx in tqdm(range(num_videos)):
    vid_key = f"video{vid_idx}.mp4"
    en_caps = captions_en.get(vid_key)
    if not en_caps:
        print(f"Warning: captions for {vid_key} not found in JSON, skipping")
        continue
    hi_caps = []

    # Process captions in small batches for efficiency
    batch_size = 8
    for i in range(0, len(en_caps), batch_size):
        batch = en_caps[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        translated = model.generate(**inputs)
        hi_batch = tokenizer.batch_decode(translated, skip_special_tokens=True)
        hi_caps.extend(hi_batch)

    captions_hi[vid_key] = hi_caps

# Save Hindi captions
with open('captions_hindi.json', 'w', encoding='utf-8') as f:
    json.dump(captions_hi, f, ensure_ascii=False, indent=4)

print("Finished Hindi caption translation and saved.")


# In[19]:


import json
import sentencepiece as spm
from collections import Counter

class VocabSPM:
    def __init__(self, captions_dict=None, freq_threshold=1, max_size=None,
                 model_prefix='hindi_spm', vocab_size=8000, character_coverage=0.9995, model_type='word'):
        #character_coverage=0.98
        self.freq_threshold = freq_threshold
        self.max_size = max_size
        self.pad_token = "<PAD>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        self.unk_token = "<UNK>"
        self.special_tokens = [self.pad_token, self.bos_token, self.eos_token, self.unk_token]

        # Build SentencePiece model on captions if captions_dict given
        if captions_dict:
            captions_file = model_prefix + '_captions.txt'
            with open(captions_file, 'w', encoding='utf-8') as f:
                for caps in captions_dict.values():
                    for c in caps:
                        f.write(c.strip() + '\n')

            spm.SentencePieceTrainer.train(
                input=captions_file,
                model_prefix=model_prefix,
                vocab_size=vocab_size,
                character_coverage=character_coverage,
                model_type=model_type,
                user_defined_symbols=self.special_tokens
            )

        # Load trained SentencePiece model
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_prefix + '.model')

        # Build word2idx and idx2word with special token offsets
        self.word2idx = {tok: idx for idx, tok in enumerate(self.special_tokens)}
        offset = len(self.special_tokens)
        for i in range(self.sp.get_piece_size()):
            piece = self.sp.id_to_piece(i)
            self.word2idx[piece] = i + offset
        self.idx2word = {idx: tok for tok, idx in self.word2idx.items()}

    def build_vocab(self, captions_dict):
        # Optional: build frequency stats across captions for filtering or stats
        counter = Counter()
        for caps in captions_dict.values():
            for c in caps:
                pieces = self.sp.encode(c, out_type=str)
                counter.update(pieces)
        # Filtering by freq_threshold and max_size if needed (optional)
        # Not strictly needed for SentencePiece vocab as vocab fixed by spm
        self.freq_counter = counter

    def numericalize(self, text):
        # Encode using SentencePiece adding special BOS/EOS tokens
        spm_pieces = self.sp.encode(text, out_type=int)  # ids from spm
        offset_pieces = [i + len(self.special_tokens) for i in spm_pieces]  # offset by specials
        return [self.word2idx[self.bos_token]] + offset_pieces + [self.word2idx[self.eos_token]]

    def decode(self, idx_list):
        # Decode token ids back to sentence, ignoring special tokens
        offset = len(self.special_tokens)
        spm_ids = [i - offset for i in idx_list if i >= offset]
        return self.sp.decode(spm_ids)


# Usage example:

CAPTIONS_FILE = '/kaggle/input/captions-hindi-1290/captions_hindi_1290.json'
with open(CAPTIONS_FILE, 'r', encoding='utf-8') as f:
    captions_hindi = json.load(f)

vocab = VocabSPM(captions_dict=captions_hindi, vocab_size=8000)
vocab.build_vocab(captions_hindi)
print("Vocabulary size (including special tokens):", len(vocab.word2idx))



# In[50]:


# -------------------------
# Cell 7: Training Loop
# -------------------------
best_bleu = 0.0
EPOCHS=15

for epoch in range(1, EPOCHS + 1):
    # ---- Training ----
    train_loss = train_one_epoch(train_loader, enc, dec, optimizer, criterion, DEVICE)
    print(f"Epoch {epoch} | Train Loss: {train_loss:.4f}")

    # ---- Validation ----
    bleu, rouge_scores, cider, _, _ = evaluate_with_metrics(val_loader, enc, dec, vocab, DEVICE, DEC_HIDDEN)
    rouge_l_f = rouge_scores['rouge-l']['f']  # Extract F1-score for ROUGE-L

    print(f"[Epoch {epoch}] Validation Metrics:")
    print(f"  BLEU-4 = {bleu:.4f}")
    print(f"  ROUGE-L (F1) = {rouge_l_f:.4f}")
    print(f"  CIDEr = {cider:.4f}")

    # ---- Save best model ----
    if bleu > best_bleu:
        best_bleu = bleu
        torch.save({
            'enc_state': enc.state_dict(),
            'dec_state': dec.state_dict(),
            'vocab': vocab.word2idx
        }, "best_checkpoint_1290_hinndi.pth")
        print(" Saved new best checkpoint.")

print(f"\nTraining complete. Best BLEU-4 achieved: {best_bleu:.4f}")


# In[59]:


import torch
import numpy as np
import os
import random
from tqdm import tqdm

# Load the best saved checkpoint

ckpt = torch.load("/kaggle/working/best_checkpoint_1290_hinndi.pth", map_location=DEVICE)
enc.load_state_dict(ckpt['enc_state'])
dec.load_state_dict(ckpt['dec_state'])
print(" Loaded checkpoint successfully.")

# If you used attention, also load it here:
# attn_refiner.load_state_dict(ckpt['attn_state'])

# =====================================
# 🔹 Evaluate on test set
# =====================================
print("Running test evaluation (BLEU, ROUGE, CIDEr)...")
test_bleu, test_rouge, test_cider ,_,_= evaluate_with_metrics(test_loader, enc, dec, vocab, DEVICE)
# Extract the ROUGE-L F1 score
test_rouge_l = test_rouge['rouge-l']['f']

print(f"\n📊 Test Metrics:")
print(f"BLEU-4  = {test_bleu:.4f}")
print(f"ROUGE-L = {test_rouge_l:.4f}")
print(f"CIDEr   = {test_cider:.4f}")


def generate_caption_for_video(video_feat, enc, dec, vocab, device, max_len=20):
    enc.eval()
    dec.eval()

    # Avoid warning about tensor creation
    if isinstance(video_feat, torch.Tensor):
        feat_tensor = video_feat.clone().detach().float().to(device)
    else:
        feat_tensor = torch.tensor(video_feat, dtype=torch.float32, device=device)

    # Ensure correct shape (B, T, D)
    if feat_tensor.dim() == 2:
        feat_tensor = feat_tensor.unsqueeze(0)   # (1, T, D)
    elif feat_tensor.dim() == 4:
        feat_tensor = feat_tensor.squeeze(0)     # (B, T, D)

    with torch.no_grad():
        encoder_outs, _ = enc(feat_tensor)

        # hidden, cell initialized same as training
        hidden = torch.zeros(1, 1, dec.lstm.hidden_size, device=device)
        cell = torch.zeros(1, 1, dec.lstm.hidden_size, device=device)

        input_word = torch.LongTensor([vocab.word2idx[vocab.bos_token]]).to(device)
        generated_tokens = []

        for _ in range(max_len):
            #  match expected shape for attention
            out, hidden, cell, attn_weights = dec.forward_step(
                input_word,
                hidden.squeeze(0),  # (B, hidden_size)
                cell.squeeze(0),    # (B, hidden_size)
                encoder_outs
            )

            # restore shape for next timestep
            hidden = hidden.unsqueeze(0)
            cell = cell.unsqueeze(0)

            next_word = out.argmax(1).item()
            if next_word == vocab.word2idx[vocab.eos_token]:
                break

            generated_tokens.append(vocab.idx2word.get(next_word, vocab.unk_token))
            input_word = torch.LongTensor([next_word]).to(device)

    return " ".join(generated_tokens)

def clean_caption(spm_text):

    # Remove SentencePiece underscore markers (replace with space)
    text = spm_text.replace("▁", " ").strip()

    # Replace <unk> tokens with placeholder or remove
    text = text.replace("<unk>", "")  # Or use "[UNK]" or any custom token

    # Remove multiple spaces
    text = ' '.join(text.split())

    return text



# In[61]:


random.seed(70)

# select random 5 videos from test_items
sample_videos = random.sample(list(test_items.keys()), min(5, len(test_items)))

for i, vid in enumerate(sample_videos):
    # Load combined feature for this video
    sample_feat = load_combined_features(vid, sample_frames=16)

    # Convert to tensor and add batch dimension
    sample_feat_tensor = torch.FloatTensor(sample_feat).unsqueeze(0).to(DEVICE)  # (1, T, D)

    # Generate caption
    generated_caption = generate_caption_for_video(sample_feat_tensor, enc, dec, vocab, DEVICE)


    cleaned_caption = clean_caption(generated_caption)


    references = test_items[vid]  # list of ground-truth captions

    print(f"\n Video {i+1}: {vid}")
    print(f" Generated Caption: {cleaned_caption}")
    print(" Reference Captions:")
    for j, ref in enumerate(references[:3]):
        print(f"  Ref {j+1}: {ref}")


# In[29]:


# -------------------------
# Cell 7: Training Loop
# -------------------------
best_bleu = 0.0

for epoch in range(1, EPOCHS + 1):
    # ---- Training ----
    train_loss = train_one_epoch(train_loader, enc, dec, optimizer, criterion, DEVICE)
    print(f"Epoch {epoch} | Train Loss: {train_loss:.4f}")

    # ---- Validation ----
    bleu, rouge_scores, cider, _, _ = evaluate_with_metrics(val_loader, enc, dec, vocab, DEVICE, DEC_HIDDEN)
    rouge_l_f = rouge_scores['rouge-l']['f']  # Extract F1-score for ROUGE-L

    print(f"[Epoch {epoch}] Validation Metrics:")
    print(f"  BLEU-4 = {bleu:.4f}")
    print(f"  ROUGE-L (F1) = {rouge_l_f:.4f}")
    print(f"  CIDEr = {cider:.4f}")

    # ---- Save best model ----
    if bleu > best_bleu:
        best_bleu = bleu
        torch.save({
            'enc_state': enc.state_dict(),
            'dec_state': dec.state_dict(),
            'vocab': vocab.word2idx
        }, "best_checkpoint_1290_3f_hindi_lstm-2.pth")
        print(" Saved new best checkpoint.")

print(f"\nTraining complete. Best BLEU-4 achieved: {best_bleu:.4f}")


# In[30]:


import torch
import numpy as np
import os
import random
from tqdm import tqdm

# =====================================
# 🔹 Load the best saved checkpoint
# =====================================
ckpt = torch.load("/kaggle/working/best_checkpoint_1290_3f_hindi_lstm-2.pth", map_location=DEVICE)
enc.load_state_dict(ckpt['enc_state'])
dec.load_state_dict(ckpt['dec_state'])
print(" Loaded checkpoint successfully.")

# If you used attention, also load it here:
# attn_refiner.load_state_dict(ckpt['attn_state'])

# =====================================
# 🔹 Evaluate on test set
# =====================================
print("Running test evaluation (BLEU, ROUGE, CIDEr)...")
test_bleu, test_rouge, test_cider ,_,_= evaluate_with_metrics(test_loader, enc, dec, vocab, DEVICE)
# Extract the ROUGE-L F1 score
test_rouge_l = test_rouge['rouge-l']['f']

print(f"\n📊 Test Metrics:")
print(f"BLEU-4  = {test_bleu:.4f}")
print(f"ROUGE-L = {test_rouge_l:.4f}")
print(f"CIDEr   = {test_cider:.4f}")


def generate_caption_for_video(video_feat, enc, dec, vocab, device, max_len=20):
    enc.eval()
    dec.eval()

    # Avoid warning about tensor creation
    if isinstance(video_feat, torch.Tensor):
        feat_tensor = video_feat.clone().detach().float().to(device)
    else:
        feat_tensor = torch.tensor(video_feat, dtype=torch.float32, device=device)

    # Ensure correct shape (B, T, D)
    if feat_tensor.dim() == 2:
        feat_tensor = feat_tensor.unsqueeze(0)   # (1, T, D)
    elif feat_tensor.dim() == 4:
        feat_tensor = feat_tensor.squeeze(0)     # (B, T, D)

    with torch.no_grad():
        encoder_outs, _ = enc(feat_tensor)

        # hidden, cell initialized same as training
        hidden = torch.zeros(1, 1, dec.lstm.hidden_size, device=device)
        cell = torch.zeros(1, 1, dec.lstm.hidden_size, device=device)

        input_word = torch.LongTensor([vocab.word2idx[vocab.bos_token]]).to(device)
        generated_tokens = []

        for _ in range(max_len):
            #  match expected shape for attention
            out, hidden, cell, attn_weights = dec.forward_step(
                input_word,
                hidden.squeeze(0),  # (B, hidden_size)
                cell.squeeze(0),    # (B, hidden_size)
                encoder_outs
            )

            # restore shape for next timestep
            hidden = hidden.unsqueeze(0)
            cell = cell.unsqueeze(0)

            next_word = out.argmax(1).item()
            if next_word == vocab.word2idx[vocab.eos_token]:
                break

            generated_tokens.append(vocab.idx2word.get(next_word, vocab.unk_token))
            input_word = torch.LongTensor([next_word]).to(device)

    return " ".join(generated_tokens)

def clean_caption(spm_text):
    """
    Cleans SentencePiece text output by:
    - Removing underscores used as word boundary markers.
    - Replacing <unk> tokens with a placeholder or removing them.
    - Stripping extra spaces.
    """
    # Remove SentencePiece underscore markers (replace with space)
    text = spm_text.replace("▁", " ").strip()

    # Replace <unk> tokens with placeholder or remove
    text = text.replace("<unk>", "")  # Or use "[UNK]" or any custom token

    # Remove multiple spaces
    text = ' '.join(text.split())

    return text



# In[31]:


random.seed(70)

# select random 5 videos from test_items
sample_videos = random.sample(list(test_items.keys()), min(5, len(test_items)))

for i, vid in enumerate(sample_videos):
    # Load combined feature for this video
    sample_feat = load_combined_features(vid, sample_frames=16)

    # Convert to tensor and add batch dimension
    sample_feat_tensor = torch.FloatTensor(sample_feat).unsqueeze(0).to(DEVICE)  # (1, T, D)

    # Generate caption
    generated_caption = generate_caption_for_video(sample_feat_tensor, enc, dec, vocab, DEVICE)


    cleaned_caption = clean_caption(generated_caption)


    references = test_items[vid]  # list of ground-truth captions

    print(f"\n Video {i+1}: {vid}")
    print(f" Generated Caption: {cleaned_caption}")
    print(" Reference Captions:")
    for j, ref in enumerate(references[:3]):
        print(f"  Ref {j+1}: {ref}")


# In[43]:


# -------------------------
# Cell 7: Training Loop
# -------------------------
best_bleu = 0.0

for epoch in range(1, EPOCHS + 1):
    # ---- Training ----
    train_loss = train_one_epoch(train_loader, enc, dec, optimizer, criterion, DEVICE)
    print(f"Epoch {epoch} | Train Loss: {train_loss:.4f}")

    # ---- Validation ----
    bleu, rouge_scores, cider, _, _ = evaluate_with_metrics(val_loader, enc, dec, vocab, DEVICE, DEC_HIDDEN)
    rouge_l_f = rouge_scores['rouge-l']['f']  # Extract F1-score for ROUGE-L

    print(f"[Epoch {epoch}] Validation Metrics:")
    print(f"  BLEU-4 = {bleu:.4f}")
    print(f"  ROUGE-L (F1) = {rouge_l_f:.4f}")
    print(f"  CIDEr = {cider:.4f}")

    # ---- Save best model ----
    if bleu > best_bleu:
        best_bleu = bleu
        torch.save({
            'enc_state': enc.state_dict(),
            'dec_state': dec.state_dict(),
            'vocab': vocab.word2idx
        }, "checkpoint_1290_3f_hindi_lstm-2-no_bi.pth")
        print(" Saved new best checkpoint.")

print(f"\nTraining complete. Best BLEU-4 achieved: {best_bleu:.4f}")


# In[35]:


import torch
import numpy as np
import os
import random
from tqdm import tqdm

# =====================================
# 🔹 Load the best saved checkpoint
# =====================================
ckpt = torch.load("/kaggle/working/checkpoint_1290_3f_hindi_lstm-2-no_bi.pth", map_location=DEVICE)
enc.load_state_dict(ckpt['enc_state'])
dec.load_state_dict(ckpt['dec_state'])
print(" Loaded checkpoint successfully.")

# If you used attention, also load it here:
# attn_refiner.load_state_dict(ckpt['attn_state'])

# =====================================
# 🔹 Evaluate on test set
# =====================================
print("Running test evaluation (BLEU, ROUGE, CIDEr)...")
test_bleu, test_rouge, test_cider ,_,_= evaluate_with_metrics(test_loader, enc, dec, vocab, DEVICE)
# Extract the ROUGE-L F1 score
test_rouge_l = test_rouge['rouge-l']['f']

print(f"\n📊 Test Metrics:")
print(f"BLEU-4  = {test_bleu:.4f}")
print(f"ROUGE-L = {test_rouge_l:.4f}")
print(f"CIDEr   = {test_cider:.4f}")


def generate_caption_for_video(video_feat, enc, dec, vocab, device, max_len=20):
    enc.eval()
    dec.eval()

    # Avoid warning about tensor creation
    if isinstance(video_feat, torch.Tensor):
        feat_tensor = video_feat.clone().detach().float().to(device)
    else:
        feat_tensor = torch.tensor(video_feat, dtype=torch.float32, device=device)

    # Ensure correct shape (B, T, D)
    if feat_tensor.dim() == 2:
        feat_tensor = feat_tensor.unsqueeze(0)   # (1, T, D)
    elif feat_tensor.dim() == 4:
        feat_tensor = feat_tensor.squeeze(0)     # (B, T, D)

    with torch.no_grad():
        encoder_outs, _ = enc(feat_tensor)

        # hidden, cell initialized same as training
        hidden = torch.zeros(1, 1, dec.lstm.hidden_size, device=device)
        cell = torch.zeros(1, 1, dec.lstm.hidden_size, device=device)

        input_word = torch.LongTensor([vocab.word2idx[vocab.bos_token]]).to(device)
        generated_tokens = []

        for _ in range(max_len):
            #  match expected shape for attention
            out, hidden, cell, attn_weights = dec.forward_step(
                input_word,
                hidden.squeeze(0),  # (B, hidden_size)
                cell.squeeze(0),    # (B, hidden_size)
                encoder_outs
            )

            # restore shape for next timestep
            hidden = hidden.unsqueeze(0)
            cell = cell.unsqueeze(0)

            next_word = out.argmax(1).item()
            if next_word == vocab.word2idx[vocab.eos_token]:
                break

            generated_tokens.append(vocab.idx2word.get(next_word, vocab.unk_token))
            input_word = torch.LongTensor([next_word]).to(device)

    return " ".join(generated_tokens)

def clean_caption(spm_text):
    """
    Cleans SentencePiece text output by:
    - Removing underscores used as word boundary markers.
    - Replacing <unk> tokens with a placeholder or removing them.
    - Stripping extra spaces.
    """
    # Remove SentencePiece underscore markers (replace with space)
    text = spm_text.replace("▁", " ").strip()

    # Replace <unk> tokens with placeholder or remove
    text = text.replace("<unk>", "")  # Or use "[UNK]" or any custom token

    # Remove multiple spaces
    text = ' '.join(text.split())

    return text



# In[36]:


random.seed(70)

# select random 5 videos from test_items
sample_videos = random.sample(list(test_items.keys()), min(5, len(test_items)))

for i, vid in enumerate(sample_videos):
    # Load combined feature for this video
    sample_feat = load_combined_features(vid, sample_frames=16)

    # Convert to tensor and add batch dimension
    sample_feat_tensor = torch.FloatTensor(sample_feat).unsqueeze(0).to(DEVICE)  # (1, T, D)

    # Generate caption
    generated_caption = generate_caption_for_video(sample_feat_tensor, enc, dec, vocab, DEVICE)


    cleaned_caption = clean_caption(generated_caption)


    references = test_items[vid]  # list of ground-truth captions

    print(f"\n Video {i+1}: {vid}")
    print(f" Generated Caption: {cleaned_caption}")
    print(" Reference Captions:")
    for j, ref in enumerate(references[:3]):
        print(f"  Ref {j+1}: {ref}")


# lstm+transformer

# In[40]:


import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor

smoothie = SmoothingFunction().method4

def evaluate_metrics(loader, enc, dec, vocab, device):
    enc.eval(); dec.eval()

    all_refs, all_preds = [], []
    pred_sentences, ref_sentences = {}, {}
    idx_counter = 0

    with torch.no_grad():
        for (global_feats, motion_feats, caps, cap_lens) in tqdm(loader):
            global_feats, motion_feats, caps = global_feats.to(device), motion_feats.to(device), caps.to(device)
            encoder_outs = enc(global_feats, motion_feats)
            batch_size = caps.size(0)

            # Greedy decoding
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

                # For pycocoevalcap
                pred_sentences[idx_counter] = [' '.join(pred_tokens)]
                ref_sentences[idx_counter] = [' '.join(ref_tokens)]
                idx_counter += 1

    # ----- Compute BLEU-4 -----
    bleu4 = corpus_bleu(all_refs, all_preds, weights=(0.25,0.25,0.25,0.25), smoothing_function=smoothie)

    # ----- Compute CIDEr -----
    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(ref_sentences, pred_sentences)

    # ----- Compute ROUGE-L -----
    rouge_scorer = Rouge()
    rouge_score, _ = rouge_scorer.compute_score(ref_sentences, pred_sentences)

    # ----- Compute METEOR -----
    meteor_scorer = Meteor()
    meteor_score, _ = meteor_scorer.compute_score(ref_sentences, pred_sentences)

    print(f"BLEU-4: {bleu4:.4f} | CIDEr: {cider_score:.4f} | ROUGE-L: {rouge_score:.4f} | METEOR: {meteor_score:.4f}")

    return bleu4, cider_score, rouge_score, meteor_score


# In[12]:


import torch
from tqdm import tqdm
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

            # -----------------------------
            # Encoder forward
            # -----------------------------
            encoder_outs, _ = enc(feats)

            # -----------------------------
            # Transformer: start token
            # -----------------------------
            input_word = torch.LongTensor(
                [vocab.word2idx[vocab.bos_token]] * B
            ).to(device)

            preds = [[] for _ in range(B)]
            max_len = caps.size(1)

            # -----------------------------
            # Greedy decoding (Transformer)
            # -----------------------------
            for t in range(1, max_len):

                if t == 1:
                    inp_seq = input_word.unsqueeze(1)   # (B, 1)
                else:
                    inp_seq = torch.cat([inp_seq, input_word.unsqueeze(1)], dim=1)

                # ---- Transformer forward ----
                out = dec(encoder_outs, inp_seq)        # (B, seq_len, vocab)

                logits = out[:, -1, :]
                top1 = logits.argmax(1)

                input_word = top1
                for i in range(B):
                    preds[i].append(top1[i].item())

            # -----------------------------
            # Convert predictions → words
            # -----------------------------
            for i in range(B):
                pred_tokens = []
                for tok in preds[i]:
                    if tok in (vocab.word2idx[vocab.pad_token],
                               vocab.word2idx[vocab.bos_token]):
                        continue
                    if tok == vocab.word2idx[vocab.eos_token]:
                        break
                    pred_tokens.append(vocab.idx2word.get(tok, vocab.unk_token))
                all_preds.append(pred_tokens)

                # Reference tokens
                ref_tokens = []
                for tok in caps[i].cpu().numpy():
                    if tok in (vocab.word2idx[vocab.pad_token],
                               vocab.word2idx[vocab.bos_token]):
                        continue
                    if tok == vocab.word2idx[vocab.eos_token]:
                        break
                    ref_tokens.append(vocab.idx2word.get(int(tok), vocab.unk_token))
                all_refs.append([ref_tokens])

    # --------------------------------
    # Compute BLEU, ROUGE, CIDEr
    # --------------------------------
    smoothie = SmoothingFunction().method4
    bleu4 = corpus_bleu(
        all_refs,
        all_preds,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothie
    )

    refs_str = [' '.join(ref[0]) for ref in all_refs]
    preds_str = [' '.join(pred) for pred in all_preds]

    rouge_scores = rouge.get_scores(preds_str, refs_str, avg=True)

    if isinstance(rouge_scores, list):
        try:
            rouge_scores = rouge_scores[0]
        except:
            rouge_scores = None

    if not isinstance(rouge_scores, dict):
        rouge_scores = {
            "rouge-1": {"f": 0.0, "p": 0.0, "r": 0.0},
            "rouge-2": {"f": 0.0, "p": 0.0, "r": 0.0},
            "rouge-l": {"f": 0.0, "p": 0.0, "r": 0.0},
        }

    # CIDEr
    cider_score, _ = cider_scorer.compute_score(
        {i: [refs_str[i]] for i in range(len(refs_str))},
        {i: [preds_str[i]] for i in range(len(preds_str))}
    )

    return bleu4, rouge_scores, cider_score, all_refs, all_preds


# In[14]:


# -------------------------
# Cell 7: Training Loop
# -------------------------
best_bleu = 0.0

for epoch in range(1, EPOCHS + 1):

    # ---- Training ----
    train_loss = train_one_epoch(train_loader, enc, dec, optimizer, criterion, DEVICE)
    print(f"Epoch {epoch} | Train Loss: {train_loss:.4f}")

    # ---- Validation ----
    bleu, rouge_scores, cider, _, _ = evaluate_with_metrics(val_loader, enc, dec, vocab, DEVICE)

    # -----------------------------------------
    # SAFE ROUGE EXTRACTION
    # -----------------------------------------
    try:
        rouge_l_f = float(rouge_scores.get("rouge-l", {}).get("f", 0.0))
    except Exception:
        rouge_l_f = 0.0

    print(f"[Epoch {epoch}] Validation Metrics:")
    print(f"  BLEU-4 = {bleu:.4f}")
    print(f"  ROUGE-L (F1) = {rouge_l_f:.4f}")
    print(f"  CIDEr = {cider:.4f}")

    # ---- Save best model ----
    if bleu > best_bleu:
        best_bleu = bleu
        torch.save({
            'enc_state': enc.state_dict(),
            'dec_state': dec.state_dict(),
            'vocab': vocab.word2idx
        }, "checkpoint_1290_3f_hindi_lstm+transformer-5.pth")
        print(" Saved new best checkpoint.")

print(f"\nTraining complete. Best BLEU-4 achieved: {best_bleu:.4f}")


# In[15]:


def generate_caption_for_video(video_feat, enc, dec, vocab, device, max_len=30):
    enc.eval()
    dec.eval()

    with torch.no_grad():
        video_feat = video_feat.unsqueeze(0).to(device)
        encoder_outs, _ = enc(video_feat)

        generated = torch.LongTensor([[vocab.word2idx[vocab.bos_token]]]).to(device)

        for _ in range(max_len):
            # ---- FIX: remove tgt_mask argument ----
            logits = dec(encoder_outs, generated)

            next_tok = logits[:, -1, :].argmax(-1)

            if next_tok.item() == vocab.word2idx[vocab.eos_token]:
                break

            generated = torch.cat([generated, next_tok.unsqueeze(0)], dim=1)

        # Convert token IDs to words
        words = [
            vocab.idx2word.get(tok.item(), vocab.unk_token)
            for tok in generated[0]
            if tok not in (vocab.word2idx[vocab.bos_token], vocab.word2idx[vocab.eos_token])
        ]

        return ' '.join(words)


# In[16]:


import torch
import numpy as np
import os
import random
from tqdm import tqdm

# =====================================
# 🔹 Load the best saved checkpoint
# =====================================
ckpt = torch.load("/kaggle/working/checkpoint_1290_3f_hindi_lstm+transformer-5.pth", map_location=DEVICE)
enc.load_state_dict(ckpt['enc_state'])
dec.load_state_dict(ckpt['dec_state'])
print(" Loaded checkpoint successfully.")

# If you used attention, also load it here:
# attn_refiner.load_state_dict(ckpt['attn_state'])

# =====================================
# 🔹 Evaluate on test set
# =====================================
print("Running test evaluation (BLEU, ROUGE, CIDEr)...")
test_bleu, test_rouge, test_cider ,_,_= evaluate_with_metrics(test_loader, enc, dec, vocab, DEVICE)
# Extract the ROUGE-L F1 score
test_rouge_l = test_rouge['rouge-l']['f']

print(f"\n📊 Test Metrics:")
print(f"BLEU-4  = {test_bleu:.4f}")
print(f"ROUGE-L = {test_rouge_l:.4f}")
print(f"CIDEr   = {test_cider:.4f}")




def clean_caption(spm_text):
    """
    Cleans SentencePiece text output by:
    - Removing underscores used as word boundary markers.
    - Replacing <unk> tokens with a placeholder or removing them.
    - Stripping extra spaces.
    """
    # Remove SentencePiece underscore markers (replace with space)
    text = spm_text.replace("▁", " ").strip()

    # Replace <unk> tokens with placeholder or remove
    text = text.replace("<unk>", "")  # Or use "[UNK]" or any custom token

    # Remove multiple spaces
    text = ' '.join(text.split())

    return text



# **Results for Transformer (3L) +bilstm enc+3f**

# In[88]:


import random
random.seed(70)

# pick 5 random videos
sample_videos = random.sample(list(test_items.keys()), min(5, len(test_items)))


def generate_caption_for_video(video_feat, enc, dec, vocab, device, max_len=25):
    enc.eval()
    dec.eval()

    with torch.no_grad():
        # ------- encode --------
        if video_feat.dim() == 2:
            video_feat = video_feat.unsqueeze(0)        # (1, T, D)
        video_feat = video_feat.to(device)

        encoder_outs, _ = enc(video_feat)

        # ------- start with <bos> -------
        input_word = torch.LongTensor(
            [vocab.word2idx[vocab.bos_token]]
        ).to(device)                                   # (1,)

        # sequence grows autoregressively
        inp_seq = input_word.unsqueeze(0)              # (1, 1)

        tokens = []

        for _ in range(max_len):

            # decoder forward exactly like evaluation
            out = dec(encoder_outs, inp_seq)           # (1, seq_len, vocab)

            # take last timestep
            logits = out[:, -1, :]
            next_tok = logits.argmax(-1).item()

            if next_tok == vocab.word2idx[vocab.eos_token]:
                break

            # collect token
            if next_tok != vocab.word2idx[vocab.bos_token]:
                tokens.append(next_tok)

            # append next token to growing sequence
            next_tok_tensor = torch.LongTensor([[next_tok]]).to(device)   # shape (1,1)
            inp_seq = torch.cat([inp_seq, next_tok_tensor], dim=1)

        # convert ids → words
        words = [vocab.idx2word.get(tok, vocab.unk_token) for tok in tokens]
        return " ".join(words)


# ----------------------------------------------------------
# Print sample predictions for 5 random videos
# ----------------------------------------------------------
for i, vid in enumerate(sample_videos):

    # load feature
    sample_feat = load_combined_features(vid, sample_frames=16)
    sample_feat_tensor = torch.FloatTensor(sample_feat).unsqueeze(0).to(DEVICE)

    # generate caption
    generated_caption = generate_caption_for_video(sample_feat_tensor, enc, dec, vocab, DEVICE)
    cleaned_caption = clean_caption(generated_caption)

    # ground truth (list of captions)
    references = test_items[vid]

    print(f"\n Video {i+1}: {vid}")
    print(f" Generated Caption: {cleaned_caption}")
    print(" Reference Captions:")
    for j, ref in enumerate(references[:3]):
        print(f"  Ref {j+1}: {ref}")


# In[99]:


import random
random.seed(90)
#34

# pick 5 random videos
sample_videos = random.sample(list(test_items.keys()), min(10, len(test_items)))


def generate_caption_for_video(video_feat, enc, dec, vocab, device, max_len=25):
    enc.eval()
    dec.eval()

    with torch.no_grad():
        # ------- encode --------
        if video_feat.dim() == 2:
            video_feat = video_feat.unsqueeze(0)        # (1, T, D)
        video_feat = video_feat.to(device)

        encoder_outs, _ = enc(video_feat)

        # ------- start with <bos> -------
        input_word = torch.LongTensor(
            [vocab.word2idx[vocab.bos_token]]
        ).to(device)                                   # (1,)

        # sequence grows autoregressively
        inp_seq = input_word.unsqueeze(0)              # (1, 1)

        tokens = []

        for _ in range(max_len):

            # decoder forward exactly like evaluation
            out = dec(encoder_outs, inp_seq)           # (1, seq_len, vocab)

            # take last timestep
            logits = out[:, -1, :]
            next_tok = logits.argmax(-1).item()

            if next_tok == vocab.word2idx[vocab.eos_token]:
                break

            # collect token
            if next_tok != vocab.word2idx[vocab.bos_token]:
                tokens.append(next_tok)

            # append next token to growing sequence
            next_tok_tensor = torch.LongTensor([[next_tok]]).to(device)   # shape (1,1)
            inp_seq = torch.cat([inp_seq, next_tok_tensor], dim=1)

        # convert ids → words
        words = [vocab.idx2word.get(tok, vocab.unk_token) for tok in tokens]
        return " ".join(words)


# ----------------------------------------------------------
# Print sample predictions for 5 random videos
# ----------------------------------------------------------
for i, vid in enumerate(sample_videos):

    # load feature
    sample_feat = load_combined_features(vid, sample_frames=16)
    sample_feat_tensor = torch.FloatTensor(sample_feat).unsqueeze(0).to(DEVICE)

    # generate caption
    generated_caption = generate_caption_for_video(sample_feat_tensor, enc, dec, vocab, DEVICE)
    cleaned_caption = clean_caption(generated_caption)

    # ground truth (list of captions)
    references = test_items[vid]

    print(f"\n Video {i+1}: {vid}")
    print(f" Generated Caption: {cleaned_caption}")
    print(" Reference Captions:")
    for j, ref in enumerate(references[:3]):
        print(f"  Ref {j+1}: {ref}")


# In[100]:


import random
random.seed(0)
#34

# pick 5 random videos
sample_videos = random.sample(list(test_items.keys()), min(10, len(test_items)))
for i, vid in enumerate(sample_videos):

    # load feature
    sample_feat = load_combined_features(vid, sample_frames=16)
    sample_feat_tensor = torch.FloatTensor(sample_feat).unsqueeze(0).to(DEVICE)

    # generate caption
    generated_caption = generate_caption_for_video(sample_feat_tensor, enc, dec, vocab, DEVICE)
    cleaned_caption = clean_caption(generated_caption)

    # ground truth (list of captions)
    references = test_items[vid]

    print(f"\n Video {i+1}: {vid}")
    print(f" Generated Caption: {cleaned_caption}")
    print(" Reference Captions:")
    for j, ref in enumerate(references[:3]):
        print(f"  Ref {j+1}: {ref}")



# **Result of Transformer(5L)**

# In[ ]:


import random
random.seed(70)

# pick 5 random videos
sample_videos = random.sample(list(test_items.keys()), min(5, len(test_items)))


def generate_caption_for_video(video_feat, enc, dec, vocab, device, max_len=25):
    enc.eval()
    dec.eval()

    with torch.no_grad():
        # ------- encode --------
        if video_feat.dim() == 2:
            video_feat = video_feat.unsqueeze(0)        # (1, T, D)
        video_feat = video_feat.to(device)

        encoder_outs, _ = enc(video_feat)

        # ------- start with <bos> -------
        input_word = torch.LongTensor(
            [vocab.word2idx[vocab.bos_token]]
        ).to(device)                                   # (1,)

        # sequence grows autoregressively
        inp_seq = input_word.unsqueeze(0)              # (1, 1)

        tokens = []

        for _ in range(max_len):

            # decoder forward exactly like evaluation
            out = dec(encoder_outs, inp_seq)           # (1, seq_len, vocab)

            # take last timestep
            logits = out[:, -1, :]
            next_tok = logits.argmax(-1).item()

            if next_tok == vocab.word2idx[vocab.eos_token]:
                break

            # collect token
            if next_tok != vocab.word2idx[vocab.bos_token]:
                tokens.append(next_tok)

            # append next token to growing sequence
            next_tok_tensor = torch.LongTensor([[next_tok]]).to(device)   # shape (1,1)
            inp_seq = torch.cat([inp_seq, next_tok_tensor], dim=1)

        # convert ids → words
        words = [vocab.idx2word.get(tok, vocab.unk_token) for tok in tokens]
        return " ".join(words)


# ----------------------------------------------------------
# Print sample predictions for 5 random videos
# ----------------------------------------------------------
for i, vid in enumerate(sample_videos):

    # load feature
    sample_feat = load_combined_features(vid, sample_frames=16)
    sample_feat_tensor = torch.FloatTensor(sample_feat).unsqueeze(0).to(DEVICE)

    # generate caption
    generated_caption = generate_caption_for_video(sample_feat_tensor, enc, dec, vocab, DEVICE)
    cleaned_caption = clean_caption(generated_caption)

    # ground truth (list of captions)
    references = test_items[vid]

    print(f"\n Video {i+1}: {vid}")
    print(f" Generated Caption: {cleaned_caption}")
    print(" Reference Captions:")
    for j, ref in enumerate(references[:3]):
        print(f"  Ref {j+1}: {ref}")


# In[38]:


import random
random.seed(450)

# pick 5 random videos
sample_videos = random.sample(list(test_items.keys()), min(10, len(test_items)))


for i, vid in enumerate(sample_videos):

    # load feature
    sample_feat = load_combined_features(vid, sample_frames=16)
    sample_feat_tensor = torch.FloatTensor(sample_feat).unsqueeze(0).to(DEVICE)

    # generate caption
    generated_caption = generate_caption_for_video(sample_feat_tensor, enc, dec, vocab, DEVICE)
    cleaned_caption = clean_caption(generated_caption)

    # ground truth (list of captions)
    references = test_items[vid]

    print(f"\n Video {i+1}: {vid}")
    print(f" Generated Caption: {cleaned_caption}")
    print(" Reference Captions:")
    for j, ref in enumerate(references[:3]):
        print(f"  Ref {j+1}: {ref}")


# In[14]:


def train_one_epoch(train_loader, encoder, decoder, optimizer, criterion, device, clip=5.0):
    encoder.train()
    decoder.train()

    running_loss = 0.0

    for feats, caps, cap_lens in tqdm(train_loader):
        feats = feats.to(device)          # (B, T_enc, D)
        caps = caps.to(device)            # (B, T_dec)


        optimizer.zero_grad()

        # ------------------------
        # 1. ENCODER
        # ------------------------
        enc_out, _ = encoder(feats)       # (B, T_enc, enc_dim)

        # ------------------------
        # 2. MASKS FOR TRANSFORMER
        # ------------------------
        T_dec = caps.size(1)
        tgt_mask = decoder._generate_square_subsequent_mask(T_dec, device)


        # if no padding in encoder features:
        memory_key_padding_mask = None

        # ------------------------
        # 3. DECODER (CROSS-ATTN)
        # ------------------------
        outputs = decoder(
            enc_out,
            caps,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )                                  # (B, T_dec, V)

        # ------------------------
        # 4. SHIFT OUTPUTS / TARGETS
        # ------------------------
        outputs = outputs[:, :-1, :].contiguous()  # predict next token
        targets = caps[:, 1:].contiguous()

        loss = criterion(
            outputs.reshape(-1, outputs.size(-1)),
            targets.reshape(-1)
        )

        # ------------------------
        # 5. BACKPROP
        # ------------------------
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)

        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(train_loader)


# In[79]:


# -------------------------
# Cell 7: Training Loop
# -------------------------
best_bleu = 0.0

for epoch in range(1, EPOCHS + 1):

    # ---- Training ----
    train_loss = train_one_epoch(train_loader, enc, dec, optimizer, criterion, DEVICE)
    print(f"Epoch {epoch} | Train Loss: {train_loss:.4f}")

    # ---- Validation ----
    bleu, rouge_scores, cider, _, _ = evaluate_with_metrics(val_loader, enc, dec, vocab, DEVICE)

    # -----------------------------------------
    # SAFE ROUGE EXTRACTION
    # -----------------------------------------
    try:
        rouge_l_f = float(rouge_scores.get("rouge-l", {}).get("f", 0.0))
    except Exception:
        rouge_l_f = 0.0

    print(f"[Epoch {epoch}] Validation Metrics:")
    print(f"  BLEU-4 = {bleu:.4f}")
    print(f"  ROUGE-L (F1) = {rouge_l_f:.4f}")
    print(f"  CIDEr = {cider:.4f}")

    # ---- Save best model ----
    if bleu > best_bleu:
        best_bleu = bleu
        torch.save({
            'enc_state': enc.state_dict(),
            'dec_state': dec.state_dict(),
            'vocab': vocab.word2idx
        }, "checkpoint_1290_3f_hindi_lstm+transformer+attention_4L.pth")
        print(" Saved new best checkpoint.")

print(f"\nTraining complete. Best BLEU-4 achieved: {best_bleu:.4f}")


# In[81]:


import torch
import numpy as np
import os
import random
from tqdm import tqdm

# =====================================
# 🔹 Load the best saved checkpoint
# =====================================
ckpt = torch.load("/kaggle/working/checkpoint_1290_3f_hindi_lstm+transformer+attention_4L.pth", map_location=DEVICE)
enc.load_state_dict(ckpt['enc_state'])
dec.load_state_dict(ckpt['dec_state'])
print(" Loaded checkpoint successfully.")

# If you used attention, also load it here:
# attn_refiner.load_state_dict(ckpt['attn_state'])

# =====================================
# 🔹 Evaluate on test set
# =====================================
print("Running test evaluation (BLEU, ROUGE, CIDEr)...")
test_bleu, test_rouge, test_cider ,_,_= evaluate_with_metrics(test_loader, enc, dec, vocab, DEVICE)
# Extract the ROUGE-L F1 score
test_rouge_l = test_rouge['rouge-l']['f']

print(f"\n📊 Test Metrics:")
print(f"BLEU-4  = {test_bleu:.4f}")
print(f"ROUGE-L = {test_rouge_l:.4f}")
print(f"CIDEr   = {test_cider:.4f}")




def clean_caption(spm_text):

    # Remove SentencePiece underscore markers (replace with space)
    text = spm_text.replace("▁", " ").strip()

    # Replace <unk> tokens with placeholder or remove
    text = text.replace("<unk>", "")  # Or use "[UNK]" or any custom token

    # Remove multiple spaces
    text = ' '.join(text.split())

    return text



# In[84]:


import random
random.seed(70)

# pick 5 random videos
sample_videos = random.sample(list(test_items.keys()), min(5, len(test_items)))


def generate_caption_for_video(video_feat, enc, dec, vocab, device, max_len=25):
    enc.eval()
    dec.eval()

    with torch.no_grad():
        # ------- encode --------
        if video_feat.dim() == 2:
            video_feat = video_feat.unsqueeze(0)        # (1, T, D)
        video_feat = video_feat.to(device)

        encoder_outs, _ = enc(video_feat)

        # ------- start with <bos> -------
        input_word = torch.LongTensor(
            [vocab.word2idx[vocab.bos_token]]
        ).to(device)                                   # (1,)

        # sequence grows autoregressively
        inp_seq = input_word.unsqueeze(0)              # (1, 1)

        tokens = []

        for _ in range(max_len):

            # decoder forward exactly like evaluation
            out = dec(encoder_outs, inp_seq)           # (1, seq_len, vocab)

            # take last timestep
            logits = out[:, -1, :]
            next_tok = logits.argmax(-1).item()

            if next_tok == vocab.word2idx[vocab.eos_token]:
                break

            # collect token
            if next_tok != vocab.word2idx[vocab.bos_token]:
                tokens.append(next_tok)

            # append next token to growing sequence
            next_tok_tensor = torch.LongTensor([[next_tok]]).to(device)   # shape (1,1)
            inp_seq = torch.cat([inp_seq, next_tok_tensor], dim=1)

        # convert ids → words
        words = [vocab.idx2word.get(tok, vocab.unk_token) for tok in tokens]
        return " ".join(words)


# ----------------------------------------------------------
# Print sample predictions for 5 random videos
# ----------------------------------------------------------
for i, vid in enumerate(sample_videos):

    # load feature
    sample_feat = load_combined_features(vid, sample_frames=16)
    sample_feat_tensor = torch.FloatTensor(sample_feat).unsqueeze(0).to(DEVICE)

    # generate caption
    generated_caption = generate_caption_for_video(sample_feat_tensor, enc, dec, vocab, DEVICE)
    cleaned_caption = clean_caption(generated_caption)

    # ground truth (list of captions)
    references = test_items[vid]

    print(f"\n Video {i+1}: {vid}")
    print(f" Generated Caption: {cleaned_caption}")
    print(" Reference Captions:")
    for j, ref in enumerate(references[:3]):
        print(f"  Ref {j+1}: {ref}")


# In[85]:


import random
random.seed(34)

# pick 5 random videos
sample_videos = random.sample(list(test_items.keys()), min(5, len(test_items)))


def generate_caption_for_video(video_feat, enc, dec, vocab, device, max_len=25):
    enc.eval()
    dec.eval()

    with torch.no_grad():
        # ------- encode --------
        if video_feat.dim() == 2:
            video_feat = video_feat.unsqueeze(0)        # (1, T, D)
        video_feat = video_feat.to(device)

        encoder_outs, _ = enc(video_feat)

        # ------- start with <bos> -------
        input_word = torch.LongTensor(
            [vocab.word2idx[vocab.bos_token]]
        ).to(device)                                   # (1,)

        # sequence grows autoregressively
        inp_seq = input_word.unsqueeze(0)              # (1, 1)

        tokens = []

        for _ in range(max_len):

            # decoder forward exactly like evaluation
            out = dec(encoder_outs, inp_seq)           # (1, seq_len, vocab)

            # take last timestep
            logits = out[:, -1, :]
            next_tok = logits.argmax(-1).item()

            if next_tok == vocab.word2idx[vocab.eos_token]:
                break

            # collect token
            if next_tok != vocab.word2idx[vocab.bos_token]:
                tokens.append(next_tok)

            # append next token to growing sequence
            next_tok_tensor = torch.LongTensor([[next_tok]]).to(device)   # shape (1,1)
            inp_seq = torch.cat([inp_seq, next_tok_tensor], dim=1)

        # convert ids → words
        words = [vocab.idx2word.get(tok, vocab.unk_token) for tok in tokens]
        return " ".join(words)


# ----------------------------------------------------------
# Print sample predictions for 5 random videos
# ----------------------------------------------------------
for i, vid in enumerate(sample_videos):

    # load feature
    sample_feat = load_combined_features(vid, sample_frames=16)
    sample_feat_tensor = torch.FloatTensor(sample_feat).unsqueeze(0).to(DEVICE)

    # generate caption
    generated_caption = generate_caption_for_video(sample_feat_tensor, enc, dec, vocab, DEVICE)
    cleaned_caption = clean_caption(generated_caption)

    # ground truth (list of captions)
    references = test_items[vid]

    print(f"\n Video {i+1}: {vid}")
    print(f" Generated Caption: {cleaned_caption}")
    print(" Reference Captions:")
    for j, ref in enumerate(references[:3]):
        print(f"  Ref {j+1}: {ref}")

