from pathlib import Path
import re

import cv2
import numpy as np
import torch

from backend.config import (
    UNET3D_CHECKPOINT_PATH,
    PATCH_DEPTH,
    PATCH_STRIDE,
    IMAGE_SIZE,
    DEFAULT_THRESHOLD,
)
from models.unet3d import UNet3D


def natural_key(path: Path):
    # Extract the last integer in a filename when present.
    match = re.search(r"(\d+)(?!.*\d)", path.stem)
    return (int(match.group(1)) if match else -1, path.name.lower())


class CTSegmentationService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    @property
    def is_loaded(self):
        return self.model is not None

    @property
    def device_name(self):
        return str(self.device)

    def load(self):
        if not UNET3D_CHECKPOINT_PATH.is_file():
            raise FileNotFoundError(f"3D U-Net checkpoint not found: {UNET3D_CHECKPOINT_PATH}")

        self.model = UNet3D(in_channels=1, out_channels=1).to(self.device)
        try:
            checkpoint = torch.load(
                UNET3D_CHECKPOINT_PATH,
                map_location=self.device,
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(UNET3D_CHECKPOINT_PATH, map_location=self.device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

    def _load_slices(self, input_dir: Path):
        paths = sorted(input_dir.glob("*.png"), key=natural_key)
        if not paths:
            raise ValueError("No PNG slices found in the uploaded ZIP.")

        images = []
        original_shapes = []
        for path in paths:
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Could not read PNG slice: {path.name}")

            original_shapes.append(image.shape)
            if image.shape != (IMAGE_SIZE, IMAGE_SIZE):
                image = cv2.resize(
                    image,
                    (IMAGE_SIZE, IMAGE_SIZE),
                    interpolation=cv2.INTER_AREA,
                )
            images.append(image.astype(np.float32) / 255.0)

        return paths, np.stack(images, axis=0), original_shapes

    def predict_folder(
        self,
        input_dir: Path,
        output_dir: Path,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        if not self.is_loaded:
            raise RuntimeError("3D U-Net has not been loaded.")

        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths, volume, original_shapes = self._load_slices(input_dir)
        depth = volume.shape[0]

        # Overlapping 8-slice patches; include the final aligned patch.
        if depth <= PATCH_DEPTH:
            starts = [0]
        else:
            starts = list(range(0, depth - PATCH_DEPTH + 1, PATCH_STRIDE))
            final_start = depth - PATCH_DEPTH
            if starts[-1] != final_start:
                starts.append(final_start)

        probability_sum = np.zeros_like(volume, dtype=np.float32)
        probability_count = np.zeros((depth, 1, 1), dtype=np.float32)

        self.model.eval()
        with torch.inference_mode():
            for start in starts:
                patch = volume[start:start + PATCH_DEPTH]
                real_depth = patch.shape[0]

                if real_depth < PATCH_DEPTH:
                    pad = np.zeros(
                        (PATCH_DEPTH - real_depth, IMAGE_SIZE, IMAGE_SIZE),
                        dtype=np.float32,
                    )
                    patch = np.concatenate([patch, pad], axis=0)

                tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(self.device)
                logits = self.model(tensor)
                probabilities = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
                probabilities = probabilities[:real_depth]

                probability_sum[start:start + real_depth] += probabilities
                probability_count[start:start + real_depth] += 1.0

        probability_volume = probability_sum / np.maximum(probability_count, 1.0)
        mask_volume = (probability_volume >= threshold).astype(np.uint8)

        # Save each slice's probability map, binary mask and overlay.
        total_positive_pixels = 0
        for index, path in enumerate(paths):
            probability_map = np.clip(probability_volume[index] * 255.0, 0, 255).astype(np.uint8)
            mask = mask_volume[index] * 255
            source = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if source.shape != (IMAGE_SIZE, IMAGE_SIZE):
                source = cv2.resize(source, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

            overlay = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
            overlay[mask_volume[index] == 1] = (0, 0, 255)

            stem = f"slice_{index:04d}"
            cv2.imwrite(str(output_dir / f"{stem}_probability.png"), probability_map)
            cv2.imwrite(str(output_dir / f"{stem}_mask.png"), mask)
            cv2.imwrite(str(output_dir / f"{stem}_overlay.png"), overlay)
            total_positive_pixels += int(mask_volume[index].sum())

        return {
            "model": "3D U-Net with dropout",
            "device": str(self.device),
            "input_slice_count": int(depth),
            "patch_depth": PATCH_DEPTH,
            "patch_stride": PATCH_STRIDE,
            "patch_count": len(starts),
            "threshold": float(threshold),
            "positive_mask_pixels": total_positive_pixels,
            "output_slice_size": [IMAGE_SIZE, IMAGE_SIZE],
            "outputs": [
                "per-slice probability maps",
                "per-slice binary masks",
                "per-slice CT overlays",
            ],
            "warning": (
                "This model was trained on nodule-centered LIDC-IDRI PNG patches. "
                "Applying it to arbitrary full-scan PNG sequences is an experimental "
                "inference workflow and requires validation; outputs are not a diagnosis."
            ),
        }
