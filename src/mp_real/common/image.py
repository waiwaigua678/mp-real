from __future__ import annotations

import numpy as np

from mp_real.policy_client import image_tools


def preprocess_image(img: np.ndarray, resize_size: int) -> np.ndarray:
    img = np.asarray(img)
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    if img.shape[-1] == 4:
        img = img[:, :, :3]
    if np.issubdtype(img.dtype, np.floating):
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return image_tools.resize_with_pad(img.astype(np.uint8, copy=False), resize_size, resize_size)
