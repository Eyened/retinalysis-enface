from pathlib import Path
from typing import Union

import numpy as np
import pydicom
from PIL import Image


def open_mask(fpath):
    if Path(fpath).suffix == ".npy":
        return np.load(fpath)

    im = Image.open(fpath)

    # if RGBA, paste into black background before converting to 'L'
    if im.mode == "RGBA":
        new_image = Image.new("RGBA", im.size, "BLACK")
        new_image.paste(im, (0, 0), im)
        im = new_image.convert("RGB")

    return np.array(im)


def open_binary_mask(fpath):
    im = open_mask(fpath)
    if len(im.shape) == 2:
        return (im > 0).astype(np.uint8).squeeze()
    else:
        return im[:, :, 0].squeeze()


def open_image_pil(path: Union[Path, str]):
    if isinstance(path, str):
        path = Path(path)
    if path.suffix == ".dcm":
        ds = pydicom.dcmread(str(path))
        img = Image.fromarray(ds.pixel_array)
    else:
        img = Image.open(str(path))
    return img


def open_image(path: Union[Path, str], dtype: Union[np.uint8, np.float32] = np.uint8):
    if Path(path).suffix == ".npy":
        im = np.load(path)
    else:
        im = np.array(open_image_pil(path), dtype=np.uint8)
    if im.dtype == np.uint8 and dtype == np.float32:
        im = (im / 255).astype(np.float32)
    if im.dtype == np.float32 and dtype == np.uint8:
        im = np.round(im * 255).astype(np.uint8)
    return im
