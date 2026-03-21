from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from rtnls_enface.loader import FundusLoader
from tests.regression_helpers import SAMPLES_DIR


@pytest.mark.plotting
def test_plot_samples_smoke() -> None:
    """Load the packaged sample and ensure plotting completes without error."""
    loader = FundusLoader.from_folder(SAMPLES_DIR)
    assert len(loader) > 0, f"No fundus samples found in {SAMPLES_DIR}"

    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
    try:
        loader[0].plot(ax=ax)
        fig.tight_layout()
    finally:
        plt.close(fig)
