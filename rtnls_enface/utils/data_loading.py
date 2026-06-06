from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image


def _read_dicom_pixel_array(path: Union[Path, str]):
    try:
        import pydicom
    except ImportError as exc:
        raise ImportError(
            "Reading DICOM images requires the optional dependency 'pydicom'. "
            "Install pydicom, or use PNG/JPEG/TIFF image inputs."
        ) from exc

    return pydicom.dcmread(str(path)).pixel_array


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
        img = Image.fromarray(_read_dicom_pixel_array(path))
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
