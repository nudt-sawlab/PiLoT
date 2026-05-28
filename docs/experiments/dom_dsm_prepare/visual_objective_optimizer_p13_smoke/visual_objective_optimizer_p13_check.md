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
| 0000.jpg | 9.6303 | 9.5887 | 0.5286 | 0.5450 | ez0_nz0_yp0p5 | 0.000 | 0.000 | 0.500 | 8 | False |

## P12 Comparison
- regression_rate: 0.0
- non_initial_accept_rate: 1.0
- mean_chamfer_improvement: 0.04157066345214844
- mean_overlap_improvement: 0.016440736878693074
- mean_num_evaluations: 8.0
- P12 fixed search used about 83 candidates/image; P13 target is fewer evaluations.

## Interpretation
- does_visual_optimizer_improve_initial: True
- does_optimizer_reduce_evaluations_vs_p12: True
- recommended_next_step: P14 should add multi-scale coarse-to-fine visual optimization or a differentiable edge-distance objective.

## Conclusion
P13 safely improves the batch with fewer evaluations than P12.
