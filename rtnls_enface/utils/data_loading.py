from pathlib import Path

import numpy as np
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
