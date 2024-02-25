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
