# Demo data layout (`data_demo/`)

All demo cases read from this folder. Download from
[Hugging Face — PiLoT-data](https://huggingface.co/datasets/choyaa/PiLoT-data):

```bash
# from PiLoT repo root
./scripts/download_data_demo.sh
```

Partial downloads: `./scripts/download_data_demo.sh feicuiwan` or `smbu`.

```
data_demo/
├── 3dgs_model/                 # Feicuiwan / Jadebay PLY
│   └── point_cloud.ply
├── smbu_model/                 # SMBU CityGaussian checkpoint + COLMAP sparse
│   ├── checkpoints/
│   │   └── epoch=60-step=30000.ckpt
│   └── sparse/sparse/0/
├── pretrained_model/           # PiLoT refiner weights (shared by all cases)
│   └── model@mapscape@512@Fourier.ckpt
└── query/
    ├── images/
    │   ├── 3dgs_test/        # Feicuiwan query frames
    │   └── smbu_seq2/        # SMBU query frames
    └── poses/
        ├── 3dgs_test.txt       # lon lat alt roll pitch yaw
        └── smbu_seq2.txt       # x y z pitch roll yaw (normalized)
```

## Pose formats

| File | Columns |
|------|---------|
| `3dgs_test.txt` | `name lon lat alt roll pitch yaw` |
| `smbu_seq2.txt` | `name x y z pitch roll yaw` |

## Local development note

If disk space is tight, `smbu_model/` may be a symlink to your CityGaussian
training output. The Hugging Face release ships real checkpoint files under
`data_demo/smbu_model/`.

| Variable | Default |
|----------|---------|
| `SMBU_MODEL_DIR` | `data_demo/smbu_model` |
| `CITYGAUSSIAN_ROOT` | `third_party/CityGaussian` |
