"""Reconstruction of a clipped optic disc from its visible contour.

The CFI/image cut edge is censored, not fitted as an anatomical disc boundary.
The hidden portion is an extrapolation; diagnostics are not calibrated confidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares

from rtnls_enface.base import Point


class EllipseFitError(ValueError):
    """The observed contour could not support a numerical ellipse fit."""


@dataclass
class OpticDiscEllipseFit:
    """Fitted ellipse in source-image pixels, including off-image coordinates.

    ``center`` uses Point(y, x); outline/contour arrays use (x, y). The angle
    runs from +x toward +y (clockwise on an image). Semi-axes are major, minor.
    ``visible_fraction`` includes losses outside both the CFI ROI and image.
    ``extrapolation_limit`` is a rejection flag; other flags are diagnostics.
    """

    center: Point
    semi_axes: Tuple[float, float]
    angle_degrees: float
    contour_xy: np.ndarray
    excluded_contour_xy: np.ndarray
    contour_rmse_px: float
    observed_arc_degrees: float
    visible_fraction: float
    visible_iou: float
    boundary_margin_px: float
    roundness_weight: float
    flags: Tuple[str, ...]

    @property
    def is_poor_fit(self) -> bool:
        """Require visible mask agreement, an unconstrained fit, and ovality < 1.5.

        Low contour error on a short arc cannot rescue poor filled-mask overlap.
        Visibility and arc length alone do not reject a clipped disc; accepted
        clipped fits remain available for center/distance measurements.
        """
        return (not np.isfinite(self.visible_iou) or self.visible_iou < 0.80
                or "extrapolation_limit" in self.flags
                or self.semi_axes[0] / self.semi_axes[1] >= 1.5)

    @property
    def area(self) -> float:
        return float(np.pi * np.prod(self.semi_axes))

    @property
    def equivalent_radius(self) -> float:
        """Radius of a circle with the fitted ellipse's area."""
        return float(np.sqrt(np.prod(self.semi_axes)))

    @property
    def bounding_box(self) -> Tuple[float, float, float, float]:
        """Unclipped (x_min, x_max, y_min, y_max) bounds."""
        a, b = self.semi_axes
        t = np.deg2rad(self.angle_degrees)
        dx = np.hypot(a * np.cos(t), b * np.sin(t))
        dy = np.hypot(a * np.sin(t), b * np.cos(t))
        return self.center.x - dx, self.center.x + dx, self.center.y - dy, self.center.y + dy

    def outline_xy(self, num_points: int = 360) -> np.ndarray:
        """Return a closed outline; coordinates may lie outside the image."""
        if num_points < 4:
            raise ValueError("num_points must be at least 4")
        phase = np.linspace(0, 2 * np.pi, num_points + 1)
        a, b = self.semi_axes
        t = np.deg2rad(self.angle_degrees)
        x = self.center.x + a * np.cos(phase) * np.cos(t) - b * np.sin(phase) * np.sin(t)
        y = self.center.y + a * np.cos(phase) * np.sin(t) + b * np.sin(phase) * np.cos(t)
        return np.column_stack((x, y))

    def to_mask(self, shape: Tuple[int, int], origin_yx=(0, 0)) -> np.ndarray:
        """Rasterize on a chosen canvas, whose origin is in source coordinates.

        A source-sized canvas necessarily clips the ellipse. Use ``full_mask``
        to retain the reconstructed portion beyond the source image.
        """
        y, x = np.indices(shape, dtype=float)
        x += origin_yx[1] - self.center.x
        y += origin_yx[0] - self.center.y
        t = np.deg2rad(self.angle_degrees)
        u, v = x * np.cos(t) + y * np.sin(t), -x * np.sin(t) + y * np.cos(t)
        a, b = self.semi_axes
        return (u / a) ** 2 + (v / b) ** 2 <= 1

    def full_mask(self) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Return the complete mask and its (row, column) source-space origin.

        The origin can be negative. This canvas must not be treated as aligned
        with the source image without applying the returned offset.
        """
        x0, x1, y0, y1 = self.bounding_box
        origin = (int(np.floor(y0)) - 1, int(np.floor(x0)) - 1)
        shape = (int(np.ceil(y1)) - origin[0] + 2, int(np.ceil(x1)) - origin[1] + 2)
        return self.to_mask(shape, origin), origin


def fit_disc_ellipse(mask, valid_mask, *, boundary_margin_px=5.0,
                     roundness_weight=0.02, robust_scale_px=2.0) -> OpticDiscEllipseFit:
    """Fit a robust ellipse to contour points clear of the visible boundary.

    OpenCV provides the initial ellipse. Bounded nonlinear least squares then
    minimizes approximate geometric (Sampson) distances with a soft-L1 loss.
    When points are censored, a weak log-axis-ratio penalty stabilizes short
    arcs. Its pixel scale is ``roundness_weight * retained_contour_span``;
    zero disables it. No roundness penalty is applied to complete contours.

    Five pixels of boundary clearance is a tunable segmentation/CFI alignment
    allowance, not a biological threshold. Bounds on center/axes prevent runaway
    extrapolation. Fits reaching these limits, spanning less than 180 degrees,
    or matching the observed mask poorly are flagged. ``is_poor_fit`` rejects
    visible IoU below 0.80, an extrapolation limit, or major/minor axis ratio
    >= 1.5; callers decide whether to retain the candidate. Short arcs and contour error remain diagnostics.
    Raises EllipseFitError for unusable contours or numerical failures; invalid
    API arguments raise ValueError.
    """
    for name, value in (("boundary_margin_px", boundary_margin_px),
                        ("roundness_weight", roundness_weight),
                        ("robust_scale_px", robust_scale_px)):
        if not np.isfinite(value) or value < 0 or (name == "robust_scale_px" and value == 0):
            raise ValueError(f"{name} must be finite and {'positive' if name == 'robust_scale_px' else 'non-negative'}")
    mask = np.asarray(mask, dtype=bool)
    valid = np.asarray(valid_mask, dtype=bool)
    if mask.ndim != 2 or valid.shape != mask.shape:
        raise ValueError("valid_mask must have the same 2D shape as the disc mask")

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise EllipseFitError("Cannot fit an empty optic-disc mask")
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
    # Padding includes the image edge even when the ROI is all True.
    clearance = ndimage.distance_transform_edt(np.pad(valid, 1))[1:-1, 1:-1] - 1.0
    keep = clearance[contour[:, 1], contour[:, 0]] > boundary_margin_px
    points = contour[keep].astype(float)
    excluded = contour[~keep].astype(float)
    if len(points) < 12 or np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
        raise EllipseFitError("Too little curved contour remains away from the visible boundary")

    span = float(np.ptp(points, axis=0).max())
    if span < 4:
        raise EllipseFitError("Visible optic-disc contour is too small for ellipse fitting")
    # Normalize position and scale for reliable numerical optimization.
    origin = points.mean(axis=0)
    p = (points - origin) / span
    try:
        (cx, cy), (width, height), angle = cv2.fitEllipse(p.astype(np.float32))
    except cv2.error as exc:
        raise EllipseFitError("Visible contour does not support an ellipse fit") from exc
    initial = np.array([cx, cy, width / 2, height / 2, np.deg2rad(angle)])
    lower = np.r_[p.min(axis=0) - 1, 0.05, 0.05, -4 * np.pi]
    upper = np.r_[p.max(axis=0) + 1, 2.0, 2.0, 4 * np.pi]
    if not np.isfinite(initial).all():
        raise EllipseFitError("Visible contour produced a non-finite ellipse")
    initial = np.clip(initial, lower + 1e-6, upper - 1e-6)

    def distances(parameters):
        x, y, a, b, theta = parameters
        z = p - [x, y]
        u = z[:, 0] * np.cos(theta) + z[:, 1] * np.sin(theta)
        v = -z[:, 0] * np.sin(theta) + z[:, 1] * np.cos(theta)
        gradient = 2 * np.maximum(np.sqrt(u * u / a ** 4 + v * v / b ** 4), 1e-12)
        return (u * u / a ** 2 + v * v / b ** 2 - 1) / gradient

    def residuals(parameters):
        residual = distances(parameters)
        if len(excluded) and roundness_weight:
            penalty = roundness_weight * np.log(parameters[2] / parameters[3])
            # Repeated residuals scale the prior with point count and robust loss.
            residual = np.r_[residual, np.full(len(points), penalty)]
        return residual

    try:
        result = least_squares(residuals, initial, bounds=(lower, upper),
                               loss="soft_l1", f_scale=robust_scale_px / span, max_nfev=1000)
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
        raise EllipseFitError("Optic-disc ellipse optimization failed") from exc
    if not result.success or not np.isfinite(result.x).all():
        raise EllipseFitError("Optic-disc ellipse optimization did not converge")
    cx, cy, a, b, theta = result.x
    a, b = a * span, b * span
    if a < b:
        a, b, theta = b, a, theta + np.pi / 2
    center = origin + result.x[:2] * span
    z = points - center
    u = z[:, 0] * np.cos(theta) + z[:, 1] * np.sin(theta)
    v = -z[:, 0] * np.sin(theta) + z[:, 1] * np.cos(theta)
    phases = np.sort(np.mod(np.arctan2(v / b, u / a), 2 * np.pi))
    arc = float(np.rad2deg(2 * np.pi - np.diff(np.r_[phases, phases[0] + 2 * np.pi]).max()))
    rmse = float(np.sqrt(np.mean((distances(result.x) * span) ** 2)))
    fit = OpticDiscEllipseFit(Point(float(center[1]), float(center[0])),
                             (float(a), float(b)), float(np.rad2deg(theta) % 180),
                             points, excluded, rmse, arc, 0.0, 0.0,
                             float(boundary_margin_px), float(roundness_weight), ())
    # Count the full ellipse on a local canvas, including pixels outside the image.
    complete, (oy, ox) = fit.full_mask()
    h, w = mask.shape
    ys, xs = np.nonzero(complete)
    sy, sx = ys + oy, xs + ox
    inside = (sy >= 0) & (sy < h) & (sx >= 0) & (sx < w)
    fit.visible_fraction = float(np.count_nonzero(valid[sy[inside], sx[inside]]) / len(ys))
    predicted = fit.to_mask(mask.shape) & valid
    observed = mask & valid
    union = np.count_nonzero(predicted | observed)
    fit.visible_iou = float(np.count_nonzero(predicted & observed) / union) if union else 0.0
    flags = []
    if arc < 180:
        flags.append("limited_visible_arc")
    if rmse > max(2.0, 0.05 * b):
        flags.append("poor_contour_fit")
    if fit.visible_iou < 0.85:
        flags.append("low_visible_iou")
    if np.any(result.active_mask) or np.any(np.minimum(result.x - lower, upper - result.x) < 1e-3):
        flags.append("extrapolation_limit")
    fit.flags = tuple(flags)
    return fit
