# Feicuiwan / Jadebay demo (Case 3)

Localization on a partial Jadebay scene using a **vanilla 3DGS PLY** map.
Poses are in **ECEF / WGS84** (longitude, latitude, altitude).

## Prerequisites

- `pilot` conda env (see [install.md](../install.md))
- `data_demo/` from the release package (PLY + query images + pretrained refiner)

## Run

```bash
conda activate pilot

./scripts/run_feicuiwan.sh
./scripts/run_feicuiwan.sh 3dgs_test --viz
```

Equivalent:

```bash
python main.py -c configs/demos/feicuiwan.yaml --name 3dgs_test
```

## Outputs

- `outputs/3dgs_test.txt` — estimated poses (`lon lat alt roll pitch yaw`)
- With `--viz`: frames under `outputs/3dgs_test/`

## Notes

- Renderer: `render_config.type: 3dgs` (Inria diff-gaussian-rasterization)
- No CityGaussian dependency for this case
- Scene reconstructed from [ThermalGS / TSDN](https://github.com/porcofly/ThermalGS-and-TSDN-Dataset) data
