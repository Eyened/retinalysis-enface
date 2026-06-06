from __future__ import annotations

from typing import Any


DEFAULT_FUNDUSPREP_INSTALL_HINT = "pip install 'retinalysis-enface[fundusprep]'"


def _missing_fundusprep_error(install_hint: str) -> ImportError:
    return ImportError(
        "retinalysis-fundusprep is required to build an ROI mask from bounds "
        f"metadata. Install it with `{install_hint}` or pass roi_mask directly."
    )


def _load_cfi_bounds_class(install_hint: str):
    try:
        from rtnls_fundusprep.cfi_bounds import CFIBounds
    except ImportError as exc:
        raise _missing_fundusprep_error(install_hint) from exc
    return CFIBounds


def coerce_bounds(bounds: Any, install_hint: str = DEFAULT_FUNDUSPREP_INSTALL_HINT):
    if bounds is None:
        return None

    if hasattr(bounds, "make_binary_mask"):
        return bounds

    if isinstance(bounds, dict):
        CFIBounds = _load_cfi_bounds_class(install_hint)
        return CFIBounds(**bounds)

    return None


def make_roi_mask_from_bounds(
    bounds: Any,
    target_diameter: int | None = None,
    install_hint: str = DEFAULT_FUNDUSPREP_INSTALL_HINT,
):
    bounds = coerce_bounds(bounds, install_hint=install_hint)
    if bounds is None:
        return None

    if target_diameter is not None:
        transform = bounds.get_cropping_transform(target_diameter)
        bounds = bounds.warp(transform)

    return bounds.make_binary_mask()
