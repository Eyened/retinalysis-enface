from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from rtnls_enface.grids.circle import CircleField
from rtnls_enface.grids.disc_centered import (
    DiscCenteredHemifield,
    DiscCenteredQuadrant,
    DiscCenteredRing,
)
from rtnls_enface.grids.ellipse import EllipseField
from rtnls_enface.grids.etdrs import ETDRSQuadrant, ETDRSRing, ETDRSSubfield
from rtnls_enface.grids.hemifields import HemifieldField
from rtnls_enface.grids.od_only import ODOnlyField
from rtnls_enface.grids.specifications import (
    CircleGridSpecification,
    ETDRSGridSpecification,
    GridFieldSpecification,
)
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


def test_grid_description_defaults_and_overrides() -> None:
    default = ETDRSGridSpecification()
    custom = ETDRSGridSpecification(description="a custom publication grid")

    assert default.description == default.default_description
    assert custom.description == "a custom publication grid"
    assert default == custom
    assert hash(default) == hash(custom)
    assert "description" not in custom.init_kwargs()


def test_grid_field_description_composition_preserves_enum_values() -> None:
    grid = CircleGridSpecification(
        description=(
            "a circular grid centered on the fovea with radius equal to "
            "the optic-disc–fovea distance"
        )
    )
    field = GridFieldSpecification(grid, CircleField.Superior)

    assert CircleField.Superior.value == "superior"
    assert CircleField.Superior.friendly_name == "superior region"
    assert field.description == (
        "the superior region of a circular grid centered on the fovea with "
        "radius equal to the optic-disc–fovea distance"
    )


def test_all_grid_fields_have_explicit_friendly_names() -> None:
    field_enums = (
        CircleField,
        DiscCenteredHemifield,
        DiscCenteredRing,
        DiscCenteredQuadrant,
        EllipseField,
        ETDRSQuadrant,
        ETDRSRing,
        ETDRSSubfield,
        HemifieldField,
        ODOnlyField,
    )

    for enum_cls in field_enums:
        for member in enum_cls:
            assert isinstance(member.value, str)
            assert member.friendly_name
            assert "_" not in member.friendly_name
