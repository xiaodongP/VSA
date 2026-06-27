# ABC Surface Adaptation Plan

This note records how Martin and Reif's ABC-surface idea maps to the current Quadric VSA trimmed B-spline pipeline.

## Paper Summary

The paper "Trimmed Spline Surfaces with Accurate Boundary Control" proposes ABC-surfaces:

```text
a = (w b + sum_l w_l r_l(kappa_l)) / (w + sum_l w_l)
```

where:

- `b` is a base tensor-product spline surface for the global shape.
- `r_l` are ribbon surfaces near boundary segments.
- `kappa_l = [p_l, q_l]` reparametrizes the base domain into ribbon coordinates.
- `w` fades out the base at all boundaries.
- `w_l` fades out all ribbons except the ribbon assigned to the local boundary segment.

The important engineering idea is that the ordinary trimmed surface `S(trim_uv)` does not guarantee accurate 3D boundary control. In ABC-surfaces, the ribbon dominates near the boundary, so the boundary curve can be exactly the spatial curve `r_l(u,0)`.

## Current Project Mapping

Existing modules already provide most upstream inputs:

- Quadric VSA face sets and region ids: `main.cpp`, `partitioning.*`, `quadric_proxy.*`.
- Authoritative shared 3D boundaries: `trimmed_region_input.*`.
- Labeling and side segmentation: `trimmed_labeling.*`.
- Parameterization and rectangular extension: `constrained_arap_parameterization.*`, `rotation_angle_parameterization.*`, `rectangular_domain_extension.*`.
- Base tensor-product surface fitting: `tensor_product_bspline_fitter.*`.
- Reusable trimmed asset and tessellation: `reusable_trimmed_bspline_surface.*`.

## Implemented MVP

The current implementation is an ABC-inspired preview, not a full NURBS-convertible ABC surface.

Important viewer semantics:

- `Run standard trimmed B-spline pipeline` computes the trimmed B-spline reproduction pipeline and
  renders `bspline_trimmed_surface.obj` by default.
- `bspline_trimmed_surface.obj` is the standard trimmed tensor-product B-spline result. It is not an
  ABC surface.
- `bspline_trimmed_surface_abc_preview.obj` and `abc_boundary_ribbon_surfaces.obj` are experimental
  ABC-inspired debug previews exported alongside the standard result.
- The viewer only shows those experimental ABC files when an `Experimental ABC ...` render mode or
  button is selected.

Implemented:

- `trimmed_bspline_asset.json` stores:
  - base tensor-product B-spline surface;
  - normalized UV trim loops;
  - authoritative 3D boundary polylines as `spatial_polyline`;
  - G0 `boundary_ribbons`, including the ribbon B-spline surface control grids.
- `bspline_trimmed_surface_asset.obj` samples the ordinary trimmed asset.
- `bspline_trimmed_surface_abc_preview.obj` samples a boundary-controlled preview:
  - evaluate base surface in the interior;
  - find the nearest trim-boundary projection in UV;
  - evaluate the corresponding G0 `BoundaryRibbonSurface`;
  - blend the ribbon surface into the base surface inside a narrow ribbon width.
  - the default ribbon influence width is deliberately narrow, about a quarter of one sampling
    interval, so boundary vertices are controlled without collapsing the first interior row.
- `abc_boundary_ribbon_strips.obj` exports explicit G0 boundary ribbon strips:
  - first row equals the authoritative 3D boundary;
  - inner rows move inward in UV and fade the boundary residual into the base surface.
- `abc_boundary_ribbon_report.json` reports base-surface error along the authoritative boundary.
- `BoundaryRibbonSurface` and `build_g0_boundary_ribbon_surfaces(...)` create one explicit
  G0 B-spline ribbon surface per trim loop:
  - degree `u = 1`, degree `v = 3` when four rows are available;
  - first control row is the authoritative boundary polyline;
  - inner rows follow the base surface with a fading boundary residual.
- `abc_boundary_ribbon_surfaces.obj` samples these ribbon surfaces.
- `abc_boundary_ribbon_surfaces_report.json` reports ribbon counts, degrees, control-grid sizes,
  and boundary control error.
- The reusable JSON asset now contains the same G0 ribbon surfaces as a structured
  `boundary_ribbons` array, so downstream tools do not need to reconstruct the current boundary
  control layer from debug OBJ files.
- `abc_ribbon_preview_report.json` reports the preview mode and authoritative-boundary error after
  ribbon-surface blending.
- The CDT preview sampler skips regular interior grid samples that are too close to trim
  constraints, reducing boundary sliver triangles while preserving all trim-loop constraint
  vertices.
- The CDT preview sampler also inserts two inward support rings around each trim loop. These
  non-constrained samples give the boundary band a more stable transition to the interior grid
  without moving the authoritative trim boundary.
- Auxiliary preview samples are kept separated by a minimum UV spacing to avoid near-duplicate
  ring/grid vertices that create tiny triangles and unstable viewer shading.
- In the libigl viewer, ABC preview and ribbon debug modes use double-sided display by default.
  Face-based normals are useful for diagnostics, but made back-facing ABC views appear black in
  the current viewer lighting setup.
- `trimmed_bspline_asset_surface_trim_loops.obj` exports the 3D curve `S(trim_uv)`.
- Viewer render modes include `AssetOnly` and `ABCPreviewOnly`.
- Polyscope loads the ordinary asset mesh, the ABC preview, G0 ribbon strips, and G0 ribbon
  B-spline surfaces for side-by-side inspection.

## Remaining Work Toward Full Paper Reproduction

1. Boundary Ribbon Construction
   - Split the current per-loop G0 ribbon surfaces into one spline ribbon surface per labeled
     boundary segment.
   - Replace the degree-1 boundary direction with fitted boundary spline curves where exact
     interpolation or projection constraints are available.
   - Add optional normal or curvature rows for G1/G2 behavior.

2. Reparametrization `kappa_l`
   - Fit a 2D spline map from base UV to ribbon coordinates.
   - Enforce corner interpolation.
   - Optionally enforce tangent-space consistency at corners.

3. Weight Functions
   - Build implicit functions `q_l` whose zero set defines each trim boundary segment.
   - Construct `w` and `w_l`.
   - Add plateau behavior to localize ribbon influence and avoid high degree growth.

4. Procedural ABC Evaluation
   - Evaluate base, ribbons, reparametrizations and weights directly.
   - Export dense preview meshes and diagnostics.

5. CAD/NURBS Export Approximation
   - Partition the domain into interior and boundary stripe pieces.
   - Convert each piece to a standard trimmed NURBS-compatible representation when possible.
   - Keep this separate from the preview mesh path.

## Known Differences From The Paper

- Current MVP uses nearest-boundary ribbon-surface blending, not exact spline weights.
- Current MVP uses boundary polylines as ribbon control rows, not fitted high-order segment ribbons.
- Current MVP is a visualization/evaluation preview, not a STEP/IGES-ready NURBS representation.
- Contact order is currently G0-like at the boundary; G1/G2 require ribbon normal/curvature construction.

## Recommended Next Step

Upgrade `BoundaryRibbonSurface` objects from loop-level G0 ribbons to segment-level ABC ribbons:

```cpp
struct BoundaryRibbonSurface {
    int boundary_loop_id;
    int segment_id;
    BSplineSurface3D ribbon;
    std::vector<Eigen::Vector2d> boundary_uv;
    std::vector<Eigen::Vector3d> boundary_xyz;
};
```

Next implementation step:

- build one ribbon per labeled segment instead of per full loop;
- use fitted boundary curves and Hermite/normal rows for smoother G1-capable ribbons;
- replace nearest-boundary residual blending with procedural ribbon-surface blending and
  explicit weight functions.
