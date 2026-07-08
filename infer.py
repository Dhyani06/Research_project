"""
infer.py — Run FSRCNN on all LR images and save output images.

For each LR image it produces a side-by-side PNG:
  [LR (bicubic upscale)] | [FSRCNN output] | [HR ground truth]

Outputs are saved to  results/<image_name>/
  - lr_bicubic.png   : bicubic upscaled LR (baseline)
  - fsrcnn.png       : FSRCNN super-resolved output (full colour)
  - hr.png           : original HR ground truth
  - comparison.png   : all three side-by-side with PSNR labels
"""

import argparse
import math
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont

from fsrcnn import FSRCNN


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def calc_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def tensor_to_pil_gray(t: torch.Tensor) -> Image.Image:
    """Convert (1,1,H,W) or (1,H,W) tensor in [0,1] to a grayscale PIL image."""
    t = t.squeeze().clamp(0, 1)
    return TF.to_pil_image(t.cpu(), mode="L")


def y_to_rgb(y_pil: Image.Image, original_rgb: Image.Image) -> Image.Image:
    """
    Replace the Y channel of original_rgb (resized to match y_pil) with y_pil.
    Returns a full-colour PIL image.
    """
    target_size = y_pil.size          # (W, H)
    rgb_resized = original_rgb.resize(target_size, Image.BICUBIC)
    ycbcr = rgb_resized.convert("YCbCr")
    y, cb, cr = ycbcr.split()
    # Merge super-resolved Y with Cb/Cr from bicubic-upscaled LR
    merged = Image.merge("YCbCr", (y_pil, cb, cr))
    return merged.convert("RGB")


def add_label(img: Image.Image, text: str, psnr: float = None) -> Image.Image:
    """Draw a label bar at the bottom of the image."""
    bar_h = 28
    new_img = Image.new("RGB", (img.width, img.height + bar_h), (30, 30, 30))
    new_img.paste(img.convert("RGB"), (0, 0))
    draw = ImageDraw.Draw(new_img)
    label = text if psnr is None else f"{text}  PSNR: {psnr:.2f} dB"
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    draw.text((6, img.height + 4), label, fill=(255, 255, 0), font=font)
    return new_img


def make_comparison(bicubic_rgb: Image.Image, fsrcnn_rgb: Image.Image,
                    hr_rgb: Image.Image,
                    psnr_bicubic: float, psnr_fsrcnn: float) -> Image.Image:
    """Stitch three images side-by-side with labels."""
    # Resize all to same height (HR is reference)
    h = hr_rgb.height
    def fit(img):
        ratio = h / img.height
        return img.resize((int(img.width * ratio), h), Image.BICUBIC)

    panels = [
        add_label(fit(bicubic_rgb), "Bicubic", psnr_bicubic),
        add_label(fit(fsrcnn_rgb),  "FSRCNN",  psnr_fsrcnn),
        add_label(fit(hr_rgb),      "HR (GT)"),
    ]
    total_w = sum(p.width for p in panels)
    canvas = Image.new("RGB", (total_w, panels[0].height), (0, 0, 0))
    x = 0
    for p in panels:
        canvas.paste(p, (x, 0))
        x += p.width
    return canvas


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="FSRCNN inference — produces comparison images.")
    p.add_argument("--hr_dir",     type=str, default="Dataset/HR")
    p.add_argument("--lr_dir",     type=str, default="Dataset/LR")
    p.add_argument("--scale",      type=int, default=2)
    p.add_argument("--checkpoint", type=str, default="models/fsrcnn.pth")
    p.add_argument("--out_dir",    type=str, default="results")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- load model --------------------------------------------------------
    model = FSRCNN(scale_factor=args.scale, num_channels=1).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # ---- find paired images ------------------------------------------------
    hr_paths = sorted(Path(args.hr_dir).glob(f"*_SRF_{args.scale}_HR.png"))
    lr_map   = {p.stem[:-3]: p
                for p in Path(args.lr_dir).glob(f"*_SRF_{args.scale}_LR.png")}

    if not hr_paths:
        print(f"No HR images found in {args.hr_dir} for scale {args.scale}.")
        return

    out_root = Path(args.out_dir)
    all_psnr_fsrcnn  = []
    all_psnr_bicubic = []

    print(f"\nProcessing {len(hr_paths)} image(s)...\n")

    with torch.no_grad():
        for hr_path in hr_paths:
            key = hr_path.stem[:-3]          # e.g. "img_003_SRF_2"
            if key not in lr_map:
                print(f"  [SKIP] No LR pair for {hr_path.name}")
                continue

            lr_path = lr_map[key]
            out_dir = out_root / key
            out_dir.mkdir(parents=True, exist_ok=True)

            # -- load images -------------------------------------------------
            hr_rgb = Image.open(hr_path).convert("RGB")
            lr_rgb = Image.open(lr_path).convert("RGB")

            hr_y = hr_rgb.convert("YCbCr").split()[0]   # PIL "L"
            lr_y = lr_rgb.convert("YCbCr").split()[0]

            hr_w, hr_h = hr_y.size

            # -- bicubic upscale ---------------------------------------------
            bicubic_y_pil = lr_y.resize((hr_w, hr_h), Image.BICUBIC)
            bicubic_rgb   = y_to_rgb(bicubic_y_pil, lr_rgb)

            # -- FSRCNN upscale ----------------------------------------------
            lr_tensor = TF.to_tensor(lr_y).unsqueeze(0).to(device)   # (1,1,h,w)
            pred      = model(lr_tensor)
            pred      = pred[:, :, :hr_h, :hr_w]                      # exact crop

            fsrcnn_y_pil = tensor_to_pil_gray(pred)
            fsrcnn_rgb   = y_to_rgb(fsrcnn_y_pil, lr_rgb)

            # -- PSNR (Y-channel, paper standard) ----------------------------
            hr_tensor      = TF.to_tensor(hr_y).unsqueeze(0).to(device)
            bicubic_tensor = TF.to_tensor(bicubic_y_pil).unsqueeze(0).to(device)

            psnr_fsrcnn  = calc_psnr(pred,          hr_tensor)
            psnr_bicubic = calc_psnr(bicubic_tensor, hr_tensor)

            all_psnr_fsrcnn.append(psnr_fsrcnn)
            all_psnr_bicubic.append(psnr_bicubic)

            # -- save individual images --------------------------------------
            bicubic_rgb.save(out_dir / "lr_bicubic.png")
            fsrcnn_rgb.save(out_dir  / "fsrcnn.png")
            hr_rgb.save(out_dir      / "hr.png")

            # -- save side-by-side comparison --------------------------------
            comp = make_comparison(bicubic_rgb, fsrcnn_rgb, hr_rgb,
                                   psnr_bicubic, psnr_fsrcnn)
            comp.save(out_dir / "comparison.png")

            print(f"  {hr_path.name}")
            print(f"    Bicubic PSNR : {psnr_bicubic:.2f} dB")
            print(f"    FSRCNN  PSNR : {psnr_fsrcnn:.2f} dB  "
                  f"({'▲ better' if psnr_fsrcnn > psnr_bicubic else '▼ worse'})")
            print(f"    Saved  -> {out_dir}/")

    if all_psnr_fsrcnn:
        avg_f = sum(all_psnr_fsrcnn)  / len(all_psnr_fsrcnn)
        avg_b = sum(all_psnr_bicubic) / len(all_psnr_bicubic)
        print(f"\n{'='*50}")
        print(f"Average Bicubic PSNR : {avg_b:.2f} dB")
        print(f"Average FSRCNN  PSNR : {avg_f:.2f} dB")
        diff = avg_f - avg_b
        print(f"Gain over Bicubic    : {diff:+.2f} dB")
        print(f"{'='*50}")
        print(f"\nAll results saved under: {out_root.resolve()}/")


if __name__ == "__main__":
    main()
