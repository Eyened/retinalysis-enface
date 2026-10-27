# Retinalysis enface geometry

## Optic-disc geometry

`OpticDisc(mask, fundus=None, size=1024)` normalizes the observed segmentation and
fits a reconstructed ellipse once. When attached to a fundus, its CFI ROI is used
to exclude cut edges; otherwise fitting accounts for the image edges alone.
Coordinates and fitting parameters are in the normalized mask's pixels.

- `disc.ellipse`: accepted `OpticDiscEllipseFit`, with center (`Point(y, x)`),
  semi-axes `(major, minor)` and orientation in degrees from +x toward +y.
- `disc.center`: reconstructed center, potentially outside the image.
- `disc.circle`: area-equivalent circle at that center, radius `sqrt(a*b)`.
- `temporal_point`, `nasal_point`, `superior_point`, `inferior_point`: intersections
  of directional rays with the ellipse, relative to the reconstructed disc–fovea axis.
- `closest_point(point)`: nearest ellipse boundary point, including for queries
  inside the ellipse. `distance_to_center(point)` uses the reconstructed center.
- `mask`, `center_of_mass`, bounding boxes, `boundary_pixels`, containment and
  boundary-distance utilities still describe the observed segmentation. Vessel
  exclusion uses this observed mask.

Defaults are a 5 px cut-edge margin, roundness weight 0.02 on censored contours,
and robust-loss scale 2 px. A fit is rejected when visible IoU is non-finite or
below 0.80, when it reaches an extrapolation limit, or when ovality (major/minor
semi-axis ratio) is >= 1.5. The strict ovality limit applies to complete and clipped
discs alike; no area or area/ODFD² acceptance threshold is applied. Low contour error cannot
rescue a fit that disagrees with the visible segmentation.
Insufficient usable contour, numerical failure and rejected fits raise
`InvalidDiscMaskException`; invalid API arguments raise `ValueError`.
Fundus/VascX loaders catch invalid masks and set `disc=None` as before.

Diagnostics remain inspectable: `visible_iou` is overlap with the observed mask
within the CFI, `contour_rmse_px` is RMS approximate geometric error on the retained
contour, `observed_arc_degrees` is the fitted angular span of that contour, and
`visible_fraction` is the fitted ellipse area inside the CFI and image divided by
its complete raster area. The `extrapolation_limit` flag rejects a fit; other
flags remain informational. Short arcs, visible fraction and off-image centers
alone do not reject a fit. Visible IoU compares only the ellipse and segmentation
inside the image and CFI, allowing well-supported clipped discs to remain usable.

`disc.fit_ellipse(...)` computes a separately validated fit with optional parameters;
it does not replace `disc.ellipse` or change measurements. `outline_xy()` returns a
closed outline. `to_mask(shape)` clips to that canvas; `full_mask()` returns the
complete ellipse mask plus its source-space `(row, column)` origin, which may be
negative. Preserve this offset when comparing reconstructed masks with CFI bounds.

Fundus grids, laterality and disc–fovea distances use the reconstructed geometry.
Reload using `fundus.load_disc(...)` to clear dependent geometry and vessel caches.
