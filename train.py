import argparse
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF

from dataset import FSRCNNDataset
from fsrcnn import FSRCNN, init_weights


class PairTransform:
    def __init__(self, lr_crop_size: int = 32, scale: int = 2):
        self.lr_crop_size = lr_crop_size
        self.scale = scale

    def __call__(self, hr: Image.Image, lr: Image.Image):
        # hr and lr are PIL Images (single-channel Y)
        hr_w, hr_h = hr.size
        lr_w, lr_h = lr.size

        # Verify that HR size matches scale * LR size. If not, resize HR to match scale * LR
        if hr_w != lr_w * self.scale or hr_h != lr_h * self.scale:
            hr = hr.resize((lr_w * self.scale, lr_h * self.scale), Image.BICUBIC)
            hr_w, hr_h = hr.size

        # If the image is smaller than the target crop size, resize both images
        if lr_w < self.lr_crop_size or lr_h < self.lr_crop_size:
            new_lr_w = max(self.lr_crop_size, lr_w)
            new_lr_h = max(self.lr_crop_size, lr_h)
            lr = lr.resize((new_lr_w, new_lr_h), Image.BICUBIC)
            hr = hr.resize((new_lr_w * self.scale, new_lr_h * self.scale), Image.BICUBIC)
            lr_w, lr_h = lr.size

        # Choose top-left coordinate for aligned random cropping
        x = random.randint(0, lr_w - self.lr_crop_size)
        y = random.randint(0, lr_h - self.lr_crop_size)

        lr_crop = lr.crop((x, y, x + self.lr_crop_size, y + self.lr_crop_size))
        hr_crop = hr.crop(
            (
                x * self.scale,
                y * self.scale,
                (x + self.lr_crop_size) * self.scale,
                (y + self.lr_crop_size) * self.scale,
            )
        )

        # Aligned random horizontal flip
        if random.random() > 0.5:
            lr_crop = TF.hflip(lr_crop)
            hr_crop = TF.hflip(hr_crop)

        # Aligned random vertical flip
        if random.random() > 0.5:
            lr_crop = TF.vflip(lr_crop)
            hr_crop = TF.vflip(hr_crop)

        return TF.to_tensor(hr_crop), TF.to_tensor(lr_crop)


def calc_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Compute PSNR between pred and target tensors in [0, 1].
    Matches the paper's metric: computed on Y-channel, peak value = 1.0.
    """
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def evaluate(model: nn.Module, hr_dir: str, lr_dir: str, scale: int, device: torch.device):
    """
    Evaluate PSNR on full images (Y-channel) and compare against bicubic upsampling.
    This is the correct evaluation protocol matching the FSRCNN paper.
    """
    model.eval()
    hr_paths = sorted(Path(hr_dir).glob(f"*_SRF_{scale}_HR.png"))
    lr_paths = sorted(Path(lr_dir).glob(f"*_SRF_{scale}_LR.png"))

    # Build a map from stem key to LR path
    lr_map = {}
    for p in lr_paths:
        key = p.stem[:-3]  # strip trailing "_LR"
        lr_map[key] = p

    fsrcnn_psnrs = []
    bicubic_psnrs = []

    with torch.no_grad():
        for hr_path in hr_paths:
            key = hr_path.stem[:-3]  # strip trailing "_HR"
            if key not in lr_map:
                continue

            # Load Y channels only
            hr_y = Image.open(hr_path).convert("YCbCr").split()[0]
            lr_y = Image.open(lr_map[key]).convert("YCbCr").split()[0]

            hr_tensor = TF.to_tensor(hr_y).unsqueeze(0).to(device)   # (1,1,H,W)

            # --- FSRCNN prediction ---
            lr_tensor = TF.to_tensor(lr_y).unsqueeze(0).to(device)   # (1,1,h,w)
            pred = model(lr_tensor)
            # Crop to exact HR size in case of rounding differences
            pred = pred[:, :, : hr_tensor.shape[2], : hr_tensor.shape[3]]
            fsrcnn_psnrs.append(calc_psnr(pred, hr_tensor))

            # --- Bicubic baseline ---
            bicubic_up = lr_y.resize(hr_y.size, Image.BICUBIC)
            bicubic_tensor = TF.to_tensor(bicubic_up).unsqueeze(0).to(device)
            bicubic_tensor = bicubic_tensor[:, :, : hr_tensor.shape[2], : hr_tensor.shape[3]]
            bicubic_psnrs.append(calc_psnr(bicubic_tensor, hr_tensor))

    avg_fsrcnn = sum(fsrcnn_psnrs) / len(fsrcnn_psnrs) if fsrcnn_psnrs else 0.0
    avg_bicubic = sum(bicubic_psnrs) / len(bicubic_psnrs) if bicubic_psnrs else 0.0
    return avg_fsrcnn, avg_bicubic, len(fsrcnn_psnrs)


def parse_args():
    parser = argparse.ArgumentParser(description="Train an FSRCNN super-resolution model.")
    parser.add_argument("--hr_dir", type=str, default="Dataset/HR")
    parser.add_argument("--lr_dir", type=str, default="Dataset/LR")
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint", type=str, default="models/fsrcnn.pth")
    return parser.parse_args()


def collate_fn(batch):
    hr_list, lr_list = zip(*batch)
    return torch.stack(hr_list, dim=0), torch.stack(lr_list, dim=0)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = PairTransform(lr_crop_size=32, scale=args.scale)
    dataset = FSRCNNDataset(args.hr_dir, args.lr_dir, scale=args.scale, transform=transform)

    if len(dataset) == 0:
        print(f"Error: No paired samples found for scale {args.scale} in {args.hr_dir} and {args.lr_dir}.")
        print("Please check your dataset directories and file naming convention.")
        return

    print(f"Found {len(dataset)} paired training samples.")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # num_channels=1: train on Y-channel only, matching the paper
    model = FSRCNN(scale_factor=args.scale, num_channels=1).to(device)
    init_weights(model)

    criterion = nn.MSELoss()  # Paper uses MSE loss
    optimizer = optim.Adam(
        [
            {"params": model.first.parameters(), "lr": args.lr},
            {"params": model.first_act.parameters(), "lr": args.lr},
            {"params": model.shrink.parameters(), "lr": args.lr},
            {"params": model.shrink_act.parameters(), "lr": args.lr},
            {"params": model.map_layers.parameters(), "lr": args.lr},
            {"params": model.map_act.parameters(), "lr": args.lr},
            {"params": model.expand.parameters(), "lr": args.lr},
            {"params": model.expand_act.parameters(), "lr": args.lr},
            # Paper uses 10x smaller lr for the deconv layer
            {"params": model.deconv.parameters(), "lr": args.lr * 0.1},
        ]
    )

    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_psnr = 0.0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (hr, lr) in enumerate(dataloader):
            hr = hr.to(device)
            lr = lr.to(device)

            optimizer.zero_grad()
            preds = model(lr)
            loss = criterion(preds, hr)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)

        # Evaluate on full images every epoch using the correct Y-channel PSNR
        fsrcnn_psnr, bicubic_psnr, n_eval = evaluate(
            model, args.hr_dir, args.lr_dir, args.scale, device
        )
        print(
            f"Epoch [{epoch+1}/{args.epochs}] Loss: {avg_loss:.6f} | "
            f"FSRCNN PSNR: {fsrcnn_psnr:.2f} dB | Bicubic PSNR: {bicubic_psnr:.2f} dB "
            f"(over {n_eval} images)"
        )

        if fsrcnn_psnr > best_psnr:
            best_psnr = fsrcnn_psnr
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> Saved best checkpoint (PSNR: {best_psnr:.2f} dB)")

    print(f"\nTraining complete. Best FSRCNN PSNR: {best_psnr:.2f} dB")


if __name__ == "__main__":
    main()
