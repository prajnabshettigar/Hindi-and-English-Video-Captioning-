#!/usr/bin/env python
# coding: utf-8

# In[7]:


import json
import pandas as pd

anno_file = "dataset/data/MSRVTT/MSRVTT/annotation/MSR_VTT.json"
with open(anno_file, "r") as f:
    data = json.load(f)
# Convert annotations into DataFrame
df = pd.DataFrame(data["annotations"])


print(df.head())


# In[8]:


from collections import defaultdict
import json

captions_dict = defaultdict(list)
for _, row in df.iterrows():
    vid = row['image_id'] + ".mp4"   # ensure matches filenames
    captions_dict[vid].append(row['caption'])


with open("dataset/data/MSRVTT/MSRVTT/captions.json", "w") as f:
    json.dump(captions_dict, f)

print("Example video:", list(captions_dict.items())[1])


# Extracting Global Features from the videos

# In[ ]:


import os, cv2, torch
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image  # Needed

VIDEO_DIR = os.path.join("dataset/data/MSRVTT/MSRVTT/videos/all")
CAPTIONS_FILE = os.path.join("dataset/data/MSRVTT/MSRVTT/annotation/MSR_VTT.json") 

FEATURES_DIR = "dataset/data/MSRVTT/MSRVTT/features"
os.makedirs(FEATURES_DIR, exist_ok=True)

# Load pretrained ResNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = models.resnet152(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # remove fc
resnet.eval().to(device)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def extract_features(video_path, out_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if length <= 0:
        print(f" Skipping {video_path}, no frames found.")
        return

    idxs = np.linspace(0, length-1, num_frames).astype(int)

    feats = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue

        # Convert OpenCV frame → PIL → Tensor
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)  #  FIX
        frame = transform(frame).unsqueeze(0).to(device)

        with torch.no_grad():
            feat = resnet(frame).squeeze().cpu().numpy()
        feats.append(feat)

    cap.release()

    if feats:
        np.save(out_path, np.array(feats))
    else:
        print(f" No features extracted for {video_path}")


import glob
video_files = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.mp4")))

for v in tqdm(video_files):
    vid_name = os.path.basename(v)
    outp = os.path.join(FEATURES_DIR, vid_name.replace(".mp4",".npy"))

    if not os.path.exists(outp):
        extract_features(v, outp, num_frames=16)


# Extracting Motion Features-(512,)

# In[ ]:


import os, cv2, torch
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image
import glob


VIDEO_DIR = os.path.join("MSRVTT/videos/all")
MOTION_FEATURES_DIR = "MSRVTT/features_motion"

os.makedirs(MOTION_FEATURES_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Motion: r2plus1d_18 (3D CNN) ---
motion = models.video.r2plus1d_18(pretrained=True)
motion = torch.nn.Sequential(*list(motion.children())[:-1])
motion.eval().to(device)
transform_motion = transforms.Compose([
    transforms.Resize((112,112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                         std=[0.22803, 0.22145, 0.216989])
])


def extract_motion_features(video_path, out_path, clip_length=16):
    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if length < clip_length:
        print(f" Skipping {video_path}, not enough frames for motion features.")
        cap.release()
        return
    idxs = np.linspace(0, length-1, clip_length).astype(int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = transform_motion(frame)
        frames.append(frame)
    cap.release()
    if len(frames) < clip_length:
        print(f" Insufficient frames for motion feature extraction in {video_path}")
        return
    input_tensor = torch.stack(frames).permute(1, 0, 2, 3).unsqueeze(0).to(device)  # (B, C, T, H, W)
    with torch.no_grad():
        feat = motion(input_tensor).squeeze().cpu().numpy()
    np.save(out_path, feat)

import re

def natural_key(string_):
    # Extract numbers in the string to use for numeric sorting
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', string_)]

video_files = glob.glob(os.path.join(VIDEO_DIR, "*.mp4"))
video_files = sorted(video_files, key=natural_key)

for v in tqdm(video_files):
    vid_name = os.path.basename(v).replace(".mp4","")

    # Motion features
    outp_motion = os.path.join(MOTION_FEATURES_DIR, vid_name + ".npy")
    if not os.path.exists(outp_motion):
        extract_motion_features(v, outp_motion, clip_length=16)


# Extracting Local features-(16,49,2048)

# In[ ]:


#Local features extraction
import os
import cv2
import torch
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image
import glob

VIDEO_DIR = os.path.join("MSRVTT/videos/all")
LOCAL_FEATURES_DIR = "MSRVTT/features_local"
os.makedirs(LOCAL_FEATURES_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained ResNet152
resnet = models.resnet152(pretrained=True).to(device).eval()

# Register hook to capture output of the last conv layer (layer4)
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

resnet.layer4.register_forward_hook(get_activation('layer4'))

transform_local = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def extract_local_features_conv_maps(video_path, out_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if length <= 0:
        print(f" Skipping {video_path}, no frames found.")
        return

    idxs = np.linspace(0, length-1, num_frames).astype(int)
    feats = []

    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue

        # Convert to PIL Image and preprocess
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        inp = transform_local(frame).unsqueeze(0).to(device)

        with torch.no_grad():
            _ = resnet(inp)  # Trigger forward hook

        # Extract intermediate activations: shape (1, 2048, 7, 7)
        local_feat_map = activation['layer4'].squeeze(0).cpu().numpy()

        # Reshape to (49, 2048) as 49 local descriptors per frame
        local_features = local_feat_map.reshape(local_feat_map.shape[0], -1).T  # (49, 2048)
        feats.append(local_features)

    cap.release()

    if feats:
        # Shape: (num_frames, 49, 2048)
        np.save(out_path, np.array(feats))
    else:
        print(f" No local features extracted for {video_path}")


import re

def natural_key(string_):
    # Extract numbers in the string to use for numeric sorting
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', string_)]

video_files = glob.glob(os.path.join(VIDEO_DIR, "*.mp4"))
video_files = sorted(video_files, key=natural_key)

for v in tqdm(video_files):
    vid_name = os.path.basename(v).replace(".mp4", "")
    outp_local = os.path.join(LOCAL_FEATURES_DIR, vid_name + ".npy")
    if not os.path.exists(outp_local):
        extract_local_features_conv_maps(v, outp_local, num_frames=16)

