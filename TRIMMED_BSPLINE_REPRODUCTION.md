# Trimmed B-Spline Reproduction

This project implements a single-region baseline pipeline for reproducing the main engineering stages around Vaitkus and Varady 2018/2019.

## 2018 Modules

- Rectangular parameter-domain extension: `rectangular_domain_extension.*`.
- Polyharmonic 3D extension with optional isocurve and mesh fairness: `polyharmonic_3d_extension.*`.
- Tensor-product cubic B-spline fitting on the completed rectangle: `tensor_product_bspline_fitter.*`.
- Reusable trimmed-surface asset export: `reusable_trimmed_bspline_surface.*` stores the untrimmed tensor-product B-spline surface plus 2D trim loops.
- Trim-loop based sampled surface clipping in `trimmed_bspline_pipeline.*` and asset-derived tessellation in `reusable_trimmed_bspline_surface.*`.

## 2019 Modules

- Boundary segmentation from Quadric VSA face sets: `trimmed_region_input.*`.
- Automatic labeling with candidate rejection, virtual corners, concatenation and reduction: `trimmed_labeling.*`.
- Bezier guiding frame construction: `bezier_guiding_frame.*`.
- Constrained ARAP/KKT parameterization and residual smoothing: `constrained_arap_parameterization.*` and `rotation_angle_parameterization.*`.

## Unpublished Parameters

- Candidate rejection thresholds, weak/parallel/obtuse corner angles, labeling tie epsilon.
- Guiding-frame degree, projection sample count, frame fairness weight.
- Rectangular grid density and margin.
- Polyharmonic regularization, fairness blend, isocurve/mesh fairness blend.
- B-spline control-grid size and control-net regularization.

## Engineering Approximations

- Multiply connected rotation-angle initialization is explicitly rejected until a non-contractible loop basis is added.
- Trimmed surface output is produced by regular-grid clipping against UV trim loops; the rectangular extension itself uses CGAL CDT.
- `trimmed_bspline_asset.json` is the reusable surface representation. Viewer meshes with authoritative-boundary snapping or harmonic boundary correction are preview/debug derivatives, not the canonical trimmed B-spline asset.
- Baseline A-D share spline degree, knot type, control-grid size and regularization; their parameter domains and fitting sample sets differ by design.
- Boundary fitting is measured by evaluating the final surface at trim-loop UV vertices and comparing to authoritative 3D boundary vertices.

## Quadric VSA Integration

- Input is a Quadric VSA region face set represented by `face_region_ids` and `target_region_id`.
- Shared authoritative boundaries are extracted by `trimmed_region_input` and are not smoothed for watertight reconstruction.
- Guide positions may be smoothed for labeling/frame estimation only.
- Feature barrier and user marker hooks are carried by `BoundarySegmentationConfig`.

## Defaults

- Cubic tensor-product surface, open-uniform knots, 8x8 control grid.
- KKT ARAP residual smoothing enabled with small weight.
- G1 and G2 polyharmonic extension are both run; final result uses G2 with optional fairness.
- Output directory: `trimmed_bspline_output/`.

## Known Failures

- Regions with holes are segmented, but the current rotation-angle initialization marks multiply connected regions unsupported.
- Highly non-quad-like regions may fail admissible labeling reduction or produce weak guiding-frame confidence.
- Severe UV flips fail rather than being clamped or repaired.
- Thin sliver triangles can make the B-spline normal equations ill-conditioned.

## Ablation Outputs

- A: existing parameterization + B-spline fitting.
- B: ordinary ARAP/KKT + B-spline fitting.
- C: ARAP + 2D/3D extension + B-spline fitting.
- D: automatic labeling + constrained ARAP + extension + B-spline fitting.
- Compare `baseline_ablation.csv` and `metrics.json` for RMS/max fitting error, weak control points and condition estimate.
