import sounddevice as sd
import torch
from torch import nn
from model import MatchboxNet
from dataset import get_MFCC_transform
from data_transforms import PadTo

import json

SAMPLE_RATE = 16000
CHUNK_DURATION = 1.0
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)

# Regular PyTorch model
# MODEL_PATH = r"models/MatchboxNet_3x2x64_12_state_dict.pt"
# model = MatchboxNet(in_channels=64, n_classes=12, B=3, S=2)
# model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
# model.eval()

# TorchScript
MODEL_PATH = r"models/MatchboxNet_3x2x64_12_jit.pt"
model = torch.jit.load(MODEL_PATH)
model.eval()

with open("models/idx_to_label.json", "r") as f:
    idx_to_label_map = json.load(f)
idx_to_label_map = {int(k): v for k, v in idx_to_label_map.items()}

def extract_features(audio):
    transform = nn.Sequential(
        get_MFCC_transform(),
        PadTo(128),
    )
    return transform(audio.squeeze())

def run_inference(X, model):
    with torch.no_grad():
        logits = model(X)
        y_pred = logits.argmax().item()
        return y_pred
    
def idx_to_label(i):
    return idx_to_label_map[i]

def callback(indata, frames, time, status):
    if status:
        print(status)
    audio = torch.from_numpy(indata.T)
    X = extract_features(audio)
    y_pred = run_inference(X, model)
    print("Prediction:", idx_to_label(y_pred))

def main():
    with sd.InputStream(callback=callback, channels=1, samplerate=SAMPLE_RATE, blocksize=CHUNK_SIZE):
        print("Listening... (Press Ctrl+C to stop.)")
        while True:
            sd.sleep(100)

if __name__ == "__main__":
    main()