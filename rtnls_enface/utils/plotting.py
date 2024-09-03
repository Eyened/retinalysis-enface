from math import ceil
from typing import List

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


def plot_gridfns(plot_fns: List[None], ncols=4):
    nrows = ceil(len(plot_fns) / ncols)
    fig = plt.figure(figsize=(16, 20))
    axs = ImageGrid(
        fig,
        111,  # similar to subplot(111)
        nrows_ncols=(nrows, ncols),  # creates 2x2 grid of axes
        axes_pad=0.1,  # pad between axes in inch.
    )
    for i, fn in enumerate(plot_fns):
        fn(ax=axs[i], fig=fig)


def plot_grid(plot_fn, nrows=4, ncols=4):
    fig = plt.figure(figsize=(nrows * 4, ncols * 4))
    axs = ImageGrid(
        fig,
        111,  # similar to subplot(111)
        nrows_ncols=(nrows, ncols),  # creates 2x2 grid of axes
        axes_pad=0.1,  # pad between axes in inch.
    )

    for i, ax in enumerate(axs):
        row = i // ncols
        col = i % ncols
        plot_fn(ax=ax, row=row, col=col)


def plot_rows(plot_fn, nrows=4, ncols=2):
    fig = plt.figure(figsize=(nrows * 4, ncols * 4), dpi=300)
    axs = ImageGrid(
        fig,
        111,  # similar to subplot(111)
        nrows_ncols=(nrows, ncols),  # creates 2x2 grid of axes
        axes_pad=0.1,  # pad between axes in inch.
    )
    for i in range(nrows):
        plot_fn(axs=axs[i * ncols : (i + 1) * ncols], row=i, fig=fig)
    return fig, axs
