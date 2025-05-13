from contextlib import contextmanager
from math import ceil
from typing import List

import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


def to_longer_side(image: np.ndarray, side_length: int) -> np.ndarray:
    # Get the original dimensions of the image
    height, width = image.shape[:2]

    # Determine the scaling factor based on the longer dimension
    if height > width:
        scaling_factor = side_length / height
    else:
        scaling_factor = side_length / width

    # Calculate new dimensions while maintaining aspect ratio
    new_height = int(height * scaling_factor)
    new_width = int(width * scaling_factor)

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height))

    return resized_image


def square_and_resize(image: np.ndarray, size: int) -> np.ndarray:
    # Get the current height and width of the image
    h, w = image.shape[:2]

    # Calculate padding to make the image square
    if h > w:
        padding = ((0, 0), ((h - w) // 2, (h - w) // 2), (0, 0))  # Pad width
    else:
        padding = (((w - h) // 2, (w - h) // 2), (0, 0), (0, 0))  # Pad height

    # Apply padding
    padded_image = np.pad(image, padding, mode="constant", constant_values=128)

    # Resize the padded square image to the target size
    resized_image = cv2.resize(padded_image, (size, size))

    return resized_image


def plot_gridfns(plot_fns: List[None], ncols=4):
    nrows = ceil(len(plot_fns) / ncols)
    fig = plt.figure(figsize=(16, 20))
    axs = ImageGrid(
        fig,
        111,  # similar to subplot(111)
        nrows_ncols=(nrows, ncols),  # creates 2x2 grid of axes
        axes_pad=0.1,  # pad between axes in inch.
        share_all=True,
    )
    for i, fn in enumerate(plot_fns):
        fn(ax=axs[i])


def plot_grid(plot_fn, nrows=4, ncols=4, dpi=300):
    fig = plt.figure(figsize=(nrows * 4, ncols * 4), dpi=dpi)
    axs = ImageGrid(
        fig,
        111,  # similar to subplot(111)
        nrows_ncols=(nrows, ncols),  # creates 2x2 grid of axes
        axes_pad=0.1,  # pad between axes in inch.
        share_all=True,
    )

    axs[0].get_yaxis().set_ticks([])
    axs[0].get_xaxis().set_ticks([])

    for i, ax in enumerate(axs):
        row = i // ncols
        col = i % ncols
        plot_fn(ax=ax, row=row, col=col, i=i)

    return fig, axs


def plot_rows(plot_fn, nrows=4, ncols=2, dpi=300, subplots=False):
    if subplots:
        fig, axs = plt.subplots(
            nrows, ncols, figsize=(nrows * 4.096, ncols * 4.096), dpi=dpi
        )
        axs = axs.flat
    else:
        fig = plt.figure(figsize=(nrows * 4.096, ncols * 4.096), dpi=dpi)
        axs = ImageGrid(
            fig,
            111,  # similar to subplot(111)
            nrows_ncols=(nrows, ncols),  # creates 2x2 grid of axes
            axes_pad=0.1,  # pad between axes in inch.
            share_all=True,
        )
    for i in range(nrows):
        plot_fn(axs=axs[i * ncols : (i + 1) * ncols], row=i, fig=fig)
    return fig, axs


def plot_columns(plot_fn, nrows=4, ncols=2, dpi=300):
    fig = plt.figure(figsize=(nrows * 4.096, ncols * 4.096), dpi=dpi)
    axs = ImageGrid(
        fig,
        111,  # similar to subplot(111)
        nrows_ncols=(nrows, ncols),  # creates 2x2 grid of axes
        axes_pad=0.1,  # pad between axes in inch.
        share_all=True,
        aspect=False,
    )
    axs.set_aspect("equal")
    axs[0].get_yaxis().set_ticks([])
    axs[0].get_xaxis().set_ticks([])
    for i in range(ncols):
        plot_fn(axs=[axs[row * ncols + i] for row in range(nrows)], col=i, fig=fig)
    return fig, axs


@contextmanager
def mpl_grid(nrows, ncols, figsize=(15, 10), dpi=150):
    """
    Context manager that creates a Matplotlib grid and provides a flattened array of axes for easy plotting.

    Args:
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.
        figsize (tuple): Size of the figure (width, height).

    Yields:
        fig (matplotlib.figure.Figure): The created figure.
        axes (np.ndarray): Flattened array of axes for easy access by index.
    """
    fig = plt.figure(dpi=dpi, figsize=figsize)
    axes = ImageGrid(
        fig,
        111,  # similar to subplot(111)
        nrows_ncols=(nrows, ncols),  # creates 2x2 grid of axes
        axes_pad=0.1,  # pad between axes in inch.
        share_all=True,
        aspect=False,
    )
    # axes = axes.flatten()  # Flatten the axes array for 1D indexing
    try:
        yield fig, axes
    finally:
        plt.tight_layout()
        plt.show()
