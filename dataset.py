from pathlib import Path
from typing import List, Tuple

from PIL import Image
from torch.utils.data import Dataset


class FSRCNNDataset(Dataset):
    """Dataset that pairs HR and LR images using the dataset naming convention."""

    def __init__(self, hr_dir: str, lr_dir: str, scale: int = 2, transform=None):
        self.hr_dir = Path(hr_dir)
        self.lr_dir = Path(lr_dir)
        self.scale = scale
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self) -> List[Tuple[Path, Path]]:
        # Filter files by suffix to avoid duplicate files or intermediate generation output clashes
        hr_files = sorted(self.hr_dir.glob("*_HR.png"))
        lr_files = sorted(self.lr_dir.glob("*_LR.png"))

        hr_map = {}
        for path in hr_files:
            key = self._extract_key(path.name)
            if key:
                hr_map[key] = path

        lr_map = {}
        for path in lr_files:
            key = self._extract_key(path.name)
            if key:
                lr_map[key] = path

        pairs = []
        for key in sorted(set(hr_map) & set(lr_map)):
            if self._matches_scale(hr_map[key].name, self.scale) and self._matches_scale(lr_map[key].name, self.scale):
                pairs.append((hr_map[key], lr_map[key]))
        return pairs

    @staticmethod
    def _extract_key(name: str) -> str:
        stem = Path(name).stem
        if stem.endswith("_HR"):
            return stem[:-3]
        elif stem.endswith("_LR"):
            return stem[:-3]
        return ""

    @staticmethod
    def _matches_scale(name: str, scale: int) -> bool:
        return f"_SRF_{scale}_" in name

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        hr_path, lr_path = self.samples[idx]
        hr_image = Image.open(hr_path).convert("RGB")
        lr_image = Image.open(lr_path).convert("RGB")
        if self.transform:
            return self.transform(hr_image, lr_image)
        return hr_image, lr_image
