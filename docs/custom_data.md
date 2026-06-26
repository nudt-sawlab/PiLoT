# Preparing your own data

PiLoT expects a **query image sequence**, a **pose file** (for init + eval), a
**3D map** (rendered online), and the **pretrained refiner** checkpoint.

## Directory layout

Put everything under `data_demo/` (or any path — point the YAML at it):

```
data_demo/
├── pretrained_model/
│   └── model@mapscape@512@Fourier.ckpt    # shared; from HF download
├── <your_map>/                            # see render backend below
└── query/
    ├── images/<sequence_name>/
    │   ├── 0_0.png
    │   └── ...
    └── poses/<sequence_name>.txt
```

Image names must match the first column of the pose file (e.g. `0_0.png`).

## Pose file

One line per frame. Column order depends on coordinate system
([coordinate_systems.md](coordinate_systems.md)):

**Normalized (CityGaussian / COLMAP-style map)**

```
image_name x y z pitch roll yaw
0_0.png -3.782 -5.794 0.207 73.79 0.018 147.72
```

**ECEF (geo-referenced PLY map)**

```
image_name lon lat alt roll pitch yaw
0_0.png 114.2604 22.2078 38.89 0.0 25.0 314.99
```

- Frame 0 pose is used as the **initial prior** (unless `trust_prior_sequential: true`).
- Units: meters for translation; degrees for Euler angles.

## Camera (query)

Set `default_confs.cam_query` in your config:

```yaml
cam_query:
  max_size: 512          # query resize (longer side)
  model: "PINHOLE"
  width: 960
  height: 540
  params: [fx, fy, cx, cy]
  distortion: [k1, k2, p1, p2, k3]   # often [0,0,0,0,0] if undistorted
```

Render camera: set `render_config.render_camera: [w, h, cx, cy, fx, fy]` to
match the pinhole used for map rendering (can differ from query if you mask edges).

## Map / renderer pairing

| Your map | `render_config.type` | `coordinate_system` | Map path in config |
|----------|----------------------|---------------------|--------------------|
| CityGaussian `.ckpt` | `citygs` | `normalized` | `citygs.checkpoint`, `citygs.data_path` |
| Vanilla 3DGS `.ply` | `3dgs` | `ecef` | `gs3d.ply_path` |
| OSG / 3D Tiles (legacy) | `osg` | `ecef` | `render_config.model_path`, etc. |

Copy `configs/demos/smbu_seq2.yaml` or `feicuiwan.yaml` and edit:

```yaml
default_confs:
  dataset_path: "data_demo/query"
  sequence_name: "my_seq"
  output_name: "my_seq"
  gt_pose_path: "data_demo/query/poses/my_seq.txt"
  refine:
    coordinate_system: "normalized"   # or "ecef"
    origin: [...]                     # set from frame-0 translation
```

## Run

```bash
python main.py -c configs/demos/my_seq.yaml
# or override sequence folder name:
python main.py -c configs/demos/my_seq.yaml --name my_seq
```

## Tips

- **Black backgrounds / sky holes** (common in 3DGS): set `refine.black_bg_thresh`
  and `refine.edge_mask_ratio` (see `smbu_seq2.yaml`).
- **Sequential tracking**: keep `trust_prior_sequential: false` for 7×7 rotation
  seeds; enable `refinement.use_temporal: true` for turning sequences.
- **Evaluation**: `gt_pose_path` is used for metrics at the end; can be the same
  file as init if you have ground truth.
