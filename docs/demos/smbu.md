# SMBU demo (Case 1)

UAV localization on the SMBU campus scene (seq2) using a **CityGaussian**
checkpoint as the map. Poses are in **normalized model coordinates**
(COLMAP-style xyz).

## Prerequisites

- `pilot` conda env (see [install.md](../install.md))
- `data_demo/` populated ([data_demo/README.md](../../data_demo/README.md))
- CityGaussian cloned to `third_party/CityGaussian` (or set `CITYGAUSSIAN_ROOT`)

## Run

```bash
conda activate pilot

./scripts/run_smbu_seq2.sh
./scripts/run_smbu_seq2.sh --viz
```

Equivalent:

```bash
python main.py -c configs/demos/smbu_seq2.yaml
```

Root wrapper: `./run_smbu_seq2_citygs.sh`

## Outputs

- `outputs/smbu_seq2.txt` — estimated poses
- With `--viz`: rendered frames under `outputs/smbu_seq2/`

## Notes

- Renderer: `render_config.type: citygs`
- Temporal feature constraint enabled (`refinement.use_temporal: true`)
- Black-background and edge masks tuned for drone + pinhole render mismatch
- Initial pose: first line of `gt_pose_path`
