import os
import torch
import clip
import torchaudio
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# 1. Custom Dataset that yields (filename, PIL-image) pairs
class AudioFolderAsImageDataset(Dataset):
    def __init__(self, root_dir: str, sample_rate: int = 16000):
        self.root = root_dir
        # only grab .wav files; change extension as needed
        self.files = [f for f in os.listdir(root_dir) if f.lower().endswith(".wav")]
        self.sr = sample_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        path = os.path.join(self.root, fname)
        waveform, sr = torchaudio.load(path)  
        assert sr == self.sr, f"Expected {self.sr}Hz but got {sr}Hz"

        # Convert to Mel-spectrogram → log-dB → 0–255 image → RGB
        mel = torchaudio.transforms.MelSpectrogram(sample_rate=sr)(waveform)
        mel_db = torchaudio.transforms.AmplitudeToDB()(mel)  # (1, n_mels, time)
        arr = mel_db.squeeze(0).numpy()
        # Normalize to [0,255]
        arr = 255 * (arr - arr.min()) / (arr.max() - arr.min())
        img = Image.fromarray(arr.astype("uint8")).convert("RGB")
        return fname, img

# 2. Load CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# 3. Wrap in DataLoader
dataset = AudioFolderAsImageDataset("input/")
loader  = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

# 4. Encode everything, keeping track of filenames
filename2emb = {}
with torch.no_grad():
    for batch in loader:
        fnames, images = batch
        images = torch.stack([preprocess(im) for im in images]).to(device)
        embs   = model.encode_image(images)
        embs   = embs / embs.norm(dim=-1, keepdim=True)
        for fname, emb in zip(fnames, embs):
            filename2emb[fname] = emb.cpu()

# 5. Now filename2emb maps each "12345__actor__sound.wav" → its CLIP vector
#    You can use that dict for prompt-based retrieval or further fine-tuning.
