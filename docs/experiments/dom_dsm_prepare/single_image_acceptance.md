# DOM DSM Prepare Notes

## Environment

- Repository path: `/mnt/d/aiproject/PiLoT_work`
- Environment path: `/mnt/d/aiproject/PiLoT_work/.conda/pilot`
- PyTorch check: `2.7.1+cu118 True 11.8`
- `python main.py --help` runs successfully.

## Reference Data

- DOM: `data_caiwangcun/reference/caiwangcun_dom.tif`
- DSM: `data_caiwangcun/reference/caiwangcun_dsm.tif`
- CRS: `EPSG:32650`
- Bounds: `left=253383.72102534556, bottom=3363502.5166208195, right=255617.45091653088, top=3365689.8913650466`
- Resolution: `0.1400457612028412 m`
- DSM nodata: `-9999.0`

The reference files are local hard links and remain ignored by git.

## Single Image Acceptance

- Query image: `data_caiwangcun/query/images/exif_test/0000.jpg`
- Pose file: `data_caiwangcun/query/poses/exif_test.txt`
- Renderer-only output directory: `outputs/exif_test_single`

The first direct DJI XMP mapping used `pitch=-90, roll=180` and produced no depth hits. For the current `DOMDSMRenderer` convention, nadir rendering uses `pitch=0, roll=180`, with the XMP yaw retained. With that mapping, renderer-only acceptance at `128x72` produced a fully valid depth map with depth range `361.0..393.0 m`, and the RGB/overlay views show the same general water, road, and vegetation layout as the query.

Full PiLoT refinement has not been run in this step.
