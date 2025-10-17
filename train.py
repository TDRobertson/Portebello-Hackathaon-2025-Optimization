import argparse, json
from pathlib import Path
from typing import List,Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm.auto import tqdm, trange

# ------------------------------------------------------------------------------
# Dataset (exactly as before)
# ------------------------------------------------------------------------------

class BoxPackingDataset(Dataset):
    def __init__(self, jsonl_path: str,
                 max_height: float = 10.0,
                 max_box_dim: float = 5.0):
        self.max_h  = float(max_height)
        self.max_bd = float(max_box_dim)
        self.samples: List[Tuple[np.ndarray, np.ndarray, float]] = []
        p = Path(jsonl_path)
        if not p.exists():
            raise FileNotFoundError(f"{jsonl_path} not found")
        with p.open("r") as f:
            for ln,line in enumerate(f,1):
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON on line {ln}")
                    continue
                hm = rec.get("height_map")
                dims = rec.get("box_dims")
                ret = rec.get("return")
                if hm is None or dims is None or ret is None:
                    continue
                if len(hm)!=24 or any(len(r)!=24 for r in hm):
                    raise ValueError(f"Bad hm shape on line {ln}")
                if len(dims)!=3:
                    raise ValueError(f"Bad dims on line {ln}")
                hm_arr = np.array(hm,   dtype=np.float32)
                dim_arr= np.array(dims, dtype=np.float32)
                self.samples.append((hm_arr, dim_arr, float(ret)))
        if not self.samples:
            raise RuntimeError("No valid samples in " + jsonl_path)

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        hm, dims, ret = self.samples[idx]
        hm_t = torch.from_numpy(hm / self.max_h).unsqueeze(0)      # (1,24,24)
        w,d,h = dims / self.max_bd
        bd = torch.tensor([w,d,h],dtype=torch.float32).view(3,1,1).expand(3,24,24)
        x = torch.cat([hm_t, bd], dim=0)                          # (4,24,24)
        y = torch.tensor(ret, dtype=torch.float32)
        return x,y

# ------------------------------------------------------------------------------
# Model (as you requested)
# ------------------------------------------------------------------------------

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4,16,3,padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Flatten(),                          # 64×3×3 = 576
            nn.Linear(576,256), nn.ReLU(inplace=True),
            nn.Linear(256,1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)  # (B,)

# ------------------------------------------------------------------------------
# Training/Validation with proper tqdm usage
# ------------------------------------------------------------------------------

def train_epoch(model, loader, opt, criterion, device):
    model.train()
    running = 0.0
    bar = tqdm(loader, desc="Train ", leave=True, dynamic_ncols=True)
    for xb,yb in bar:
        xb,yb = xb.to(device), yb.to(device)
        preds = model(xb)
        loss = criterion(preds,yb)
        opt.zero_grad(); loss.backward(); opt.step()

        running += loss.item()*xb.size(0)
        bar.set_postfix(loss=f"{loss.item():.4f}")
    return running/len(loader.dataset)

def eval_epoch(model, loader, criterion, device):
    model.eval()
    running = 0.0
    bar = tqdm(loader, desc=" Val  ", leave=True, dynamic_ncols=True)
    with torch.no_grad():
        for xb,yb in bar:
            xb,yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb), yb)
            running += loss.item()*xb.size(0)
            bar.set_postfix(loss=f"{loss.item():.4f}")
    return running/len(loader.dataset)

# ------------------------------------------------------------------------------
# Main entry
# ------------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", "-d", required=True)
    p.add_argument("--batch-size","-b",type=int,default=32)
    p.add_argument("--epochs",   "-e",type=int,default=10)
    p.add_argument("--lr",       type=float,default=1e-3)
    p.add_argument("--val-split",type=float,default=0.2)
    p.add_argument("--no-cuda",  action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    ds = BoxPackingDataset(args.data)
    n_val = int(len(ds)*args.val_split)
    n_tr = len(ds)-n_val
    tr_ds, va_ds = random_split(ds,[n_tr,n_val])
    tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    va_ld = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = SimpleCNN().to(device)
    crit  = nn.MSELoss()
    opt   = optim.Adam(model.parameters(), lr=args.lr)

    # outer epoch loop with trange
    for epoch in trange(1, args.epochs+1, desc="Epochs"):
        train_loss = train_epoch(model, tr_ld, opt, crit, device)
        val_loss   = eval_epoch(model, va_ld,        crit, device)
        tqdm.write(f"Epoch {epoch}/{args.epochs}  Train Loss={train_loss:.4f}  Val Loss={val_loss:.4f}")

    torch.save(model.state_dict(), "model_final.pth")
    print("Done. Saved model_final.pth")

if __name__=="__main__":
    main()
