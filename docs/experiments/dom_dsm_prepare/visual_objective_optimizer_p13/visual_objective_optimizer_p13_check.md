# P13 Visual Objective Pose Optimizer

## Purpose
P13 replaces P12 fixed local grid search with a derivative-free optimizer driven by visual edge alignment.

## Method
- Not a grid search: pattern search / coordinate descent evaluates the center and coordinate neighbors.
- Optimizes only east, north, and yaw offsets.
- Altitude is fixed to the initial pose; pitch/roll remain [0, 180].
- Objective: edge_chamfer - overlap_weight * edge_overlap_ratio.
- Feature loss is not used as the optimization objective.
- Final selection uses strict visual gate, with fallback to initial.

## Batch Results
| Image | Initial chamfer | Selected chamfer | Initial overlap | Selected overlap | Candidate | East | North | Yaw | Evals | Safe worse? |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 0000.jpg | 9.6303 | 7.3452 | 0.5286 | 0.7696 | em6_np1_yz0 | -6.000 | 1.000 | 0.000 | 40 | False |
| 0001.JPG | 4.1782 | 4.0704 | 0.6847 | 0.7119 | ez0_nz0_ym0p25 | 0.000 | 0.000 | -0.250 | 29 | False |
| 0002.JPG | 4.7278 | 4.4858 | 0.4725 | 0.5030 | em1_nm0p5_ym0p25 | -1.000 | -0.500 | -0.250 | 37 | False |
| 0003.JPG | 4.3905 | 3.9065 | 0.6980 | 0.7583 | ep4_np0p5_yz0 | 4.000 | 0.500 | 0.000 | 39 | False |
| 0004.JPG | 2.9589 | 2.8428 | 0.8933 | 0.9259 | em2_nz0_ym0p812 | -2.000 | 0.000 | -0.812 | 40 | False |

## P12 Comparison
- regression_rate: 0.0
- non_initial_accept_rate: 1.0
- mean_chamfer_improvement: 0.6469694137573242
- mean_overlap_improvement: 0.07831244871458863
- mean_num_evaluations: 37.0
- P12 fixed search used about 83 candidates/image; P13 target is fewer evaluations.

## Interpretation
- does_visual_optimizer_improve_initial: True
- does_optimizer_reduce_evaluations_vs_p12: True
- recommended_next_step: P14 should add multi-scale coarse-to-fine visual optimization or a differentiable edge-distance objective.

## Conclusion
P13 safely improves the batch with fewer evaluations than P12.
