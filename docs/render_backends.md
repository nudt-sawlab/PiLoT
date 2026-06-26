# Render backends

The **render worker** (`main.py` → `rendering_worker`) produces `(color, depth)`
for each pose. Localization only needs:

- `color`: `uint8` RGB, shape `(H, W, 3)`
- `depth`: `float32`, shape `(H, W)`, same resolution as color

Backends are selected by `render_config.type` in the YAML config.

## Built-in backends

| `type` | Class | Module | Map format |
|--------|-------|--------|------------|
| `citygs` | `CityGaussianRenderer` | `pixloc/utils/citygs/citygs_render.py` | CityGaussian `.ckpt` + COLMAP sparse |
| `3dgs` | `GS3DRenderer` | `pixloc/utils/gs3d/gs3d_render.py` | Vanilla 3DGS `.ply` |
| `osg` | `RenderImageProcessor` | `pixloc/utils/osg/osg_render.py` | 3D Tiles / OSG (legacy, paper setup) |

Dispatch logic (simplified):

```python
# main.py rendering_worker
if render_type == "citygs":
    renderer = CityGaussianRenderer(render_config)
    color, depth = renderer.render(trans, euler)
elif render_type == "3dgs":
    renderer = GS3DRenderer(render_config)
    color, depth = renderer.render(trans, euler)
else:  # osg
    renderer = RenderImageProcessor(render_config)
    renderer.update_pose(trans, euler)
    color, depth = renderer.get_color_image(), renderer.get_depth_image()
```

Pose `trans` / `euler` format depends on `coordinate_system` — see
[coordinate_systems.md](coordinate_systems.md). Each renderer converts to its
internal camera representation.

## Adding a new backend (e.g. Blender, Unreal, custom mesh)

1. **Implement a renderer class** with a single entry point:

   ```python
   class MyRenderer:
       def __init__(self, config: dict): ...
       def render(self, trans: list, euler: list) -> tuple[np.ndarray, np.ndarray]:
           # return (color_uint8_hw3, depth_float32_hw)
           ...
   ```

2. **Register it** in `main.py` → `rendering_worker`:

   ```python
   elif self.render_type == "blender":
       from pixloc.utils.blender.blender_render import BlenderRenderer
       renderer = BlenderRenderer(self.render_config)
       color, depth = renderer.render(trans, euler)
   ```

3. **Add config block** in your YAML under `render_config` (paths, intrinsics,
   executable, scene file, etc.).

4. **Match cameras**: `render_config.render_camera: [w, h, cx, cy, fx, fy]`
   must match what the renderer outputs. Query camera is in `cam_query`; both
   should align after `max_size` resize (see `_setup_camera` in `main.py`).

5. **Depth**: must be metric depth in the same units as the map / pose system
   (used for 3D point back-projection in `back_project`).

### Blender (sketch)

- Headless render via `blender -b scene.blend -P render_script.py -- pose.json`
- Or persistent subprocess with `bpy` if Blender's Python is compatible.
- Convert PiLoT `trans`/`euler` to Blender camera matrix in your wrapper; return
  RGB + Z-buffer as depth.

### OSG / Cesium (existing)

The `osg` path is kept for compatibility with the original paper pipeline
(3D Tiles). Requires building `3DTilesRender` — not included in the public 3DGS
release. Use as reference for non-3DGS map servers.

### Unreal / Unity

Same pattern: wrapper process that accepts pose, returns PNG + depth EXR.
Consider a local HTTP or socket API to avoid spawning the engine per frame.

## Coordinate system

The renderer must interpret `trans`/`euler` consistently with
`render_config.coordinate_system` and `pixloc/utils/citygs/pose_convert.py` (or
your own conversion). Mismatch between map frame and pose file is the most
common integration bug.

## Testing a new backend

1. Render frame 0 at the init pose; visually compare to query `0_0.png`.
2. Run 2–3 frames with `trust_prior_sequential: true` to isolate renderer vs
   optimizer issues.
3. Enable `--viz` and inspect `outputs/<name>/`.
