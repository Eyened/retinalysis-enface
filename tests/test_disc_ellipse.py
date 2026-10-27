"""Known complete shapes test reconstruction independently of the fitting code."""
from types import SimpleNamespace

import numpy as np
import pytest

from rtnls_enface.disc import InvalidDiscMaskException, OpticDisc


def ellipse_mask(center_xy=(128, 128), axes=(60, 44), angle=25, size=256):
    yy, xx = np.indices((size, size))
    theta = np.deg2rad(angle)
    dx, dy = xx - center_xy[0], yy - center_xy[1]
    u = dx * np.cos(theta) + dy * np.sin(theta)
    v = -dx * np.sin(theta) + dy * np.cos(theta)
    return (u / axes[0]) ** 2 + (v / axes[1]) ** 2 <= 1


@pytest.mark.parametrize('center', [(128, 128), (14, 128), (-6, 128), (128, 14)])
def test_recovers_known_ellipse_across_image_edge(center):
    mask = ellipse_mask(center)
    disc = OpticDisc(mask, size=256)
    original_center = disc.center_of_mass.tuple
    fit = disc.fit_ellipse(roundness_weight=0)
    assert np.linalg.norm(np.array(fit.center.tuple_xy) - center) < 3
    np.testing.assert_allclose(fit.semi_axes, (60, 44), atol=3)
    assert fit.visible_iou > .94
    assert disc.center_of_mass.tuple == original_center
    np.testing.assert_array_equal(disc.mask, mask)
    if center == (128, 128):
        assert fit.visible_fraction == 1
        assert len(fit.excluded_contour_xy) == 0
    else:
        assert fit.visible_fraction < .9
        assert len(fit.excluded_contour_xy) > 0


@pytest.mark.parametrize('circular_roi', [False, True])
def test_censors_internal_cfi_boundary(circular_roi):
    yy, xx = np.indices((256, 256))
    valid = ((xx - 128) ** 2 + (yy - 128) ** 2 <= 110 ** 2) if circular_roi else xx < 224
    original = ellipse_mask((222, 128))
    disc = OpticDisc(original & valid, fundus=SimpleNamespace(roi_mask=valid), size=256)
    fit = disc.fit_ellipse(roundness_weight=0)
    assert np.linalg.norm(np.array(fit.center.tuple_xy) - (222, 128)) < 3
    np.testing.assert_allclose(fit.semi_axes, (60, 44), atol=3)
    assert .4 < fit.visible_fraction < .65
    assert fit.visible_iou > .94
    explicit = disc.fit_ellipse(valid_mask=valid, roundness_weight=0)
    np.testing.assert_allclose(fit.center.tuple, explicit.center.tuple)


def test_full_canvas_preserves_off_image_geometry_and_offset():
    disc = OpticDisc(ellipse_mask((-6, 128)), size=256)
    fit = disc.fit_ellipse(roundness_weight=0)
    mask, (oy, ox) = fit.full_mask()
    assert ox < 0
    assert mask.sum() > fit.to_mask((256, 256)).sum()
    assert abs(mask.sum() / fit.area - 1) < .01
    yy, xx = np.nonzero(mask)
    expected = np.count_nonzero((xx + ox >= 0) & (xx + ox < 256) & (yy + oy >= 0) & (yy + oy < 256)) / len(xx)
    assert fit.visible_fraction == pytest.approx(expected)
    outline = fit.outline_xy()
    np.testing.assert_allclose(outline[0], outline[-1])
    assert outline[:, 0].min() < 0
    assert fit.center.y == pytest.approx(128, abs=3)


def test_nearby_but_complete_disc_is_not_automatically_rejected():
    disc = OpticDisc(ellipse_mask((70, 128), angle=0), size=256)
    assert disc.distance_to_visible_bounds() < 25
    fit = disc.fit_ellipse()
    assert fit.visible_fraction == 1
    assert not fit.flags


def test_short_arc_is_flagged_even_when_overlap_is_good():
    disc = OpticDisc(ellipse_mask((-15, 128), axes=(60, 60)), size=256)
    fit = disc.fit_ellipse()
    assert 'limited_visible_arc' in fit.flags
    assert fit.visible_iou > .94


def test_cannot_invent_fit_without_visible_contour():
    mask = ellipse_mask()
    disc = OpticDisc(mask, size=256)
    with pytest.raises(InvalidDiscMaskException, match='Too little curved contour'):
        disc.fit_ellipse(valid_mask=np.zeros_like(mask))
    with pytest.raises(ValueError, match='same 2D shape'):
        disc.fit_ellipse(valid_mask=np.ones((128, 128)))
    for kwargs in ({'boundary_margin_px': -1}, {'roundness_weight': float('nan')}, {'robust_scale_px': 0}):
        with pytest.raises(ValueError):
            disc.fit_ellipse(**kwargs)


@pytest.mark.parametrize('iou,rmse,flags,reject', [
    (.80, 10.0, (), False),       # High contour error alone does not reject.
    (.799, 0.0, (), True),       # Excellent contour fit cannot rescue low IoU.
    (.002, .85, ('limited_visible_arc',), True),
    (.95, 0.0, ('extrapolation_limit',), True),
    (.95, 0.0, ('limited_visible_arc',), False),
    (float('nan'), 0.0, (), True),
])
def test_visible_overlap_and_extrapolation_acceptance(monkeypatch, iou, rmse, flags, reject):
    from dataclasses import replace
    import rtnls_enface.disc as module
    mask = ellipse_mask()
    fit = OpticDisc(mask, size=256).ellipse
    fit = replace(fit, visible_iou=iou, contour_rmse_px=rmse,
                  visible_fraction=.01, observed_arc_degrees=25, flags=flags)
    monkeypatch.setattr(module, 'fit_disc_ellipse', lambda *args, **kwargs: fit)
    if reject:
        with pytest.raises(InvalidDiscMaskException):
            OpticDisc(mask, size=256)
    else:
        disc = OpticDisc(mask, size=256)
        assert disc.ellipse is fit
        assert disc.center is fit.center
        assert disc.circle.center is fit.center
        assert disc.circle.r == pytest.approx(np.sqrt(np.prod(fit.semi_axes)))


@pytest.mark.parametrize('ovality,reject', [(1.499, False), (1.5, True), (1.501, True)])
@pytest.mark.parametrize('visible', [1.0, .2])
def test_strict_ovality_acceptance_and_loader(monkeypatch, ovality, reject, visible):
    from dataclasses import replace
    from rtnls_enface.fundus import Fundus
    import rtnls_enface.disc as module
    mask = ellipse_mask()
    fit = replace(OpticDisc(mask, size=256).ellipse,
                  semi_axes=(20.0 * ovality, 20.0), visible_fraction=visible)
    monkeypatch.setattr(module, 'fit_disc_ellipse', lambda *args, **kwargs: fit)
    if reject:
        with pytest.warns(UserWarning, match='Setting disc to None'):
            fundus = Fundus(disc_path_or_mask=mask, resolution=(256, 256))
        assert fundus.disc is None
    else:
        fundus = Fundus(disc_path_or_mask=mask, resolution=(256, 256),
                        fovea_location=(220, 128))
        assert fundus.disc.ellipse is fit
        assert np.isfinite(fundus.disc_fovea_distance)


def test_low_overlap_fit_is_cleared_by_fundus(monkeypatch):
    from dataclasses import replace
    from rtnls_enface.fundus import Fundus
    import rtnls_enface.disc as module
    mask = ellipse_mask()
    fit = replace(OpticDisc(mask, size=256).ellipse,
                  visible_iou=.003, contour_rmse_px=.5)
    monkeypatch.setattr(module, 'fit_disc_ellipse', lambda *args, **kwargs: fit)
    with pytest.warns(UserWarning, match='Setting disc to None'):
        fundus = Fundus(disc_path_or_mask=mask, resolution=(256, 256))
    assert fundus.disc is None


@pytest.mark.parametrize('failure', ['exception', 'nonfinite', 'nonconverged'])
def test_numerical_failure_is_an_invalid_disc(monkeypatch, failure):
    import rtnls_enface.disc_ellipse as module
    def failed(*args, **kwargs):
        if failure == 'exception':
            raise ValueError('numerical solver failure')
        return SimpleNamespace(success=failure != 'nonconverged', x=np.full(5, np.nan))
    monkeypatch.setattr(module, 'least_squares', failed)
    with pytest.raises(InvalidDiscMaskException):
        OpticDisc(ellipse_mask(), size=256)


@pytest.mark.parametrize('angle', [0, 25, 90, 145])
@pytest.mark.parametrize('fovea_x', [20, 240])
def test_directional_landmarks_and_nearest_boundary(angle, fovea_x):
    from rtnls_enface.base import Point
    fundus = SimpleNamespace(roi_mask=None, fovea_location=Point(160, fovea_x))
    disc = OpticDisc(ellipse_mask(angle=angle), size=256, fundus=fundus)
    a, b = disc.ellipse.semi_axes
    theta = np.deg2rad(disc.ellipse.angle_degrees)
    rotation = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
    temporal = np.array(fundus.fovea_location.tuple_xy) - disc.center.tuple_xy
    temporal /= np.linalg.norm(temporal)
    superior = np.array([temporal[1], -temporal[0]])
    if superior[1] > 0:
        superior = -superior
    assert disc.superior_point.y < disc.center.y
    assert disc.inferior_point.y > disc.center.y
    for point, direction in [(disc.temporal_point, temporal), (disc.nasal_point, -temporal),
                             (disc.superior_point, superior), (disc.inferior_point, -superior)]:
        offset = np.array(point.tuple_xy) - disc.center.tuple_xy
        np.testing.assert_allclose(offset / np.linalg.norm(offset), direction, atol=1e-10)
        uv = rotation.T @ offset
        assert (uv[0]/a)**2 + (uv[1]/b)**2 == pytest.approx(1)
    # Outside, inside, exactly on an axis, and at the ellipse center.
    for local in [(100, 80), (10, 15), (20, 0), (0, 0), (-100, -80), (0, 90)]:
        xy = np.array(disc.center.tuple_xy) + rotation @ local
        target = Point(xy[1], xy[0])
        closest = disc.closest_point(target)
        uv = rotation.T @ (np.array(closest.tuple_xy) - disc.center.tuple_xy)
        assert (uv[0]/a)**2 + (uv[1]/b)**2 == pytest.approx(1)
        # Dense independent outline approximates the global nearest solution.
        brute_distance = np.linalg.norm(disc.ellipse.outline_xy(50000) - xy, axis=1).min()
        assert closest.distance_to(target) == pytest.approx(brute_distance, abs=1e-5)


def test_nonelliptical_mask_is_rejected():
    yy, xx = np.indices((256, 256))
    # Deep concavities in a connected cross cannot be reconstructed as an oval.
    mask = ((abs(xx-128) < 12) & (abs(yy-128) < 85)) | ((abs(xx-128) < 85) & (abs(yy-128) < 12))
    with pytest.raises(InvalidDiscMaskException):
        OpticDisc(mask, size=256)


def test_fundus_keeps_invalid_disc_catch_and_clear(monkeypatch):
    from rtnls_enface.fundus import Fundus
    import rtnls_enface.disc_ellipse as module
    fundus = Fundus(disc_path_or_mask=ellipse_mask(), resolution=(256, 256),
                    fovea_location=(220, 128), roi_mask=np.ones((256, 256)))
    assert fundus.disc is not None
    assert fundus.disc_fovea_distance > 0
    def failed(*args, **kwargs):
        raise RuntimeError('numerical failure')
    monkeypatch.setattr(module, 'least_squares', failed)
    with pytest.warns(UserWarning, match='Setting disc to None'):
        fundus.load_disc(ellipse_mask())
    assert fundus.disc is None
    assert 'disc_fovea_distance' not in fundus.__dict__
