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
        # hr and lr are PIL Images
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
        hr_crop = hr.crop((x * self.scale, y * self.scale, (x + self.lr_crop_size) * self.scale, (y + self.lr_crop_size) * self.scale))

        # Aligned random horizontal flip
        if random.random() > 0.5:
            lr_crop = TF.hflip(lr_crop)
            hr_crop = TF.hflip(hr_crop)

        # Aligned random vertical flip
        if random.random() > 0.5:
            lr_crop = TF.vflip(lr_crop)
            hr_crop = TF.vflip(hr_crop)

        return TF.to_tensor(hr_crop), TF.to_tensor(lr_crop)


def parse_args():
    parser = argparse.ArgumentParser(description="Train an FSRCNN super-resolution model.")
    parser.add_argument("--hr_dir", type=str, default="Dataset/HR")
    parser.add_argument("--lr_dir", type=str, default="Dataset/LR")
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint", type=str, default="models/fsrcnn.pth")
    return parser.parse_args()


def collate_fn(batch):
    hr_list, lr_list = zip(*batch)
    return torch.stack(hr_list, dim=0), torch.stack(lr_list, dim=0)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Use a low-resolution crop size of 32 (meaning HR crop size will be 32 * scale)
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

    model = FSRCNN(scale_factor=args.scale).to(device)
    init_weights(model)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        running_mse = 0.0
        for batch_idx, (hr, lr) in enumerate(dataloader):
            hr = hr.to(device)
            lr = lr.to(device)

            optimizer.zero_grad()
            preds = model(lr)
            loss = criterion(preds, hr)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            with torch.no_grad():
                mse = nn.MSELoss()(preds, hr).item()
                running_mse += mse
                psnr = 10 * math.log10(1.0 / mse) if mse > 0 else float('inf')

            if (batch_idx + 1) % max(1, len(dataloader) // 5) == 0 or batch_idx == 0:
                print(
                    f"Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx+1}/{len(dataloader)}] "
                    f"Loss: {loss.item():.6f} | PSNR: {psnr:.2f} dB"
                )

        avg_loss = running_loss / len(dataloader)
        avg_mse = running_mse / len(dataloader)
        avg_psnr = 10 * math.log10(1.0 / avg_mse) if avg_mse > 0 else float('inf')
        print(f"Epoch [{epoch+1}/{args.epochs}] Average Loss: {avg_loss:.6f} | Average PSNR: {avg_psnr:.2f} dB")

    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
