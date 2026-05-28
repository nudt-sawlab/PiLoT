# P11 Local Visual-Safe Candidate Scorer Check

## Purpose
P10 tested structure weighting on raw-refined-derived candidates only. P11 rebuilds a local candidate scorer that includes local grid poses, known P3/P4 seeds, and corrected raw refined candidates under the same strict visual safe gate.

## Prior Evidence
| Source | Candidate | East | North | Alt | Yaw | Chamfer | Overlap | Feature loss | Meaning |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| local_gradient | initial | 0 | 0 | 0 | 0 | 5.4937 | 0.6293 | 0.1519 | baseline |
| local_gradient | north_plus_1m | 0 | 1 | 0 | 0 | 4.6455 | 0.6021 | 0.1546 | better chamfer |
| local_gradient | yaw_minus_1deg | 0 | 0 | 0 | -1 | 5.0441 | 0.6579 | 0.1562 | better chamfer and overlap |
| diagnosis | p3_best_overlap | -5 | 5 | 0 | 0 | 3.1489 | 1.0774 | 0.1467 | known best visual |
| diagnosis | p3_best_chamfer | -5 | 0 | 0 | 0 | 3.9349 | 0.8866 | 0.1576 | known good visual |
| diagnosis | p4_scale_025_fixed_alt | -3.302 | -0.526 | 0 | 0 | 4.5064 | 0.8012 | 0.1603 | known good local refined |

These are prior evidence only; P11 recomputes all metrics in this run.

## Candidate Generation
- Stage 1 coarse east offsets: [-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0]
- Stage 1 coarse north offsets: [-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0]
- Stage 2 yaw offsets: [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
- Known seeds included: True
- Raw refined included: True

## Gate Policy
- strict visual gate: chamfer <= initial and overlap >= initial
- final pose policy: strict_visual_gate_then_best_chamfer
- feature and weighted feature losses are diagnostic scorers only.

## Results
| Candidate | Source | Stage | East | North | Alt | Yaw | Chamfer | Overlap | Unweighted loss | Best weighted loss | Strict gate |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| coarse_em3_np0 | local_grid | coarse_translation | -3.000 | 0.000 | 0.000 | 0.000 | 2.0891 | 1.2774 | 0.107160 | 0.107160 | True |
| p3_best_chamfer__yaw_p0p5 | local_grid | yaw_refine | -5.000 | 0.000 | 0.000 | 0.500 | 2.1471 | 1.2435 | 0.118454 | 0.118275 | True |
| coarse_em5_nm1 | local_grid | coarse_translation | -5.000 | -1.000 | 0.000 | 0.000 | 2.1709 | 1.2701 | 0.114383 | 0.114383 | True |
| p3_best_chamfer | known_seed | known_seed | -5.000 | 0.000 | 0.000 | 0.000 | 2.2033 | 1.3421 | 0.118649 | 0.118649 | True |
| coarse_ep0_np3 | local_grid | coarse_translation | 0.000 | 3.000 | 0.000 | 0.000 | 2.3671 | 1.1056 | 0.120457 | 0.120457 | True |
| coarse_em5_nm1__yaw_m0p5 | local_grid | yaw_refine | -5.000 | -1.000 | 0.000 | -0.500 | 2.4058 | 1.2674 | 0.118617 | 0.118617 | True |
| coarse_em3_np0__yaw_m0p5 | local_grid | yaw_refine | -3.000 | 0.000 | 0.000 | -0.500 | 2.4463 | 1.4152 | 0.112924 | 0.112924 | True |
| coarse_em5_nm1__yaw_m2 | local_grid | yaw_refine | -5.000 | -1.000 | 0.000 | -2.000 | 2.4751 | 1.2864 | 0.139456 | 0.139456 | True |
| coarse_ep0_nm3 | local_grid | coarse_translation | 0.000 | -3.000 | 0.000 | 0.000 | 2.4761 | 1.0944 | 0.093863 | 0.093863 | True |
| coarse_em5_nm1__yaw_p0p5 | local_grid | yaw_refine | -5.000 | -1.000 | 0.000 | 0.500 | 2.4976 | 1.2375 | 0.114032 | 0.113751 | True |
| coarse_em1_nm1 | local_grid | coarse_translation | -1.000 | -1.000 | 0.000 | 0.000 | 2.5028 | 1.0814 | 0.098081 | 0.098081 | True |
| coarse_em1_np1 | local_grid | coarse_translation | -1.000 | 1.000 | 0.000 | 0.000 | 2.5354 | 1.1763 | 0.104826 | 0.104826 | True |

## Ranking Comparison
| Scorer | Top-1 candidate | Top-1 visual safe? | Spearman vs chamfer | Spearman vs overlap | Contains visual top1 in top5? |
| --- | --- | --- | ---: | ---: | --- |
| unweighted_feature | raw_refined_full | False | -0.08367354377779401 | -0.24529780564263315 | False |
| uniform | raw_refined_full | False | -0.08367354377779401 | -0.24529780564263315 | False |
| dom_edge | raw_refined_full | False | -0.07364948603922139 | -0.22721805059415323 | False |
| depth_gradient | raw_refined_full | False | -0.06450025515783331 | -0.21369468542684267 | False |
| combined | raw_refined_full | False | -0.06991324633666252 | -0.222771014070132 | False |
| low_texture_downweight | raw_refined_full | False | -0.09940220164759062 | -0.27917912079900864 | False |

## Safe Selection
- selected_by_visual: coarse_em3_np0
- selected_by_unweighted_feature: coarse_ep0_nm3
- selected_by_weighted_feature_by_mode: {"combined": "coarse_ep0_nm3", "depth_gradient": "coarse_ep0_nm3", "dom_edge": "coarse_ep0_nm3", "low_texture_downweight": "coarse_ep0_nm3", "uniform": "coarse_ep0_nm3"}
- result_pose_safe_p11.txt uses: coarse_em3_np0
- safe_output_worse_than_initial: False

## Interpretation
- local candidate set contains better than initial: True
- strict gate accepts non-initial: True
- raw refined still misses good region: True
- unweighted feature selects visual good candidate: False
- any structure weight improves rank alignment: True
- recommended next step: Use visual-safe selection or a hybrid visual-feature scorer for P12.

## Conclusion
Case B: Local visual candidate search recovers a better pose, but feature scorers remain unreliable.
