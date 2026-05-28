# P12 Batch Local Visual-Safe Candidate Search

## Purpose
P11 showed local visual-safe search improves one image. P12 applies the same fixed search policy to multiple EXIF-recovered query poses and checks generalization plus systematic offsets.

## Why Batch Is Necessary
Single-image search does not prove generalization. P12 keeps candidate generation fixed and does not tune the range for any one query image.

## Search Policy
- coarse east offsets: [-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0]
- coarse north offsets: [-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0]
- yaw refinement offsets: [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
- topK visual for yaw: 5
- enable alt refine: False
- final pose policy: strict_visual_gate_then_best_chamfer

## Gate Policy
- strict visual gate: chamfer <= initial and overlap >= initial
- final selection: lowest chamfer among strict-pass candidates; fallback initial if none pass

## Batch Results
| Image | Initial chamfer | Selected chamfer | Initial overlap | Selected overlap | Selected candidate | East | North | Yaw | Strict non-initial? | Safe worse? |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |
| 0000.jpg | 9.6303 | 7.5847 | 0.5286 | 0.7296 | coarse_em3_np0 | -3.000 | 0.000 | 0.000 | True | False |
| 0001.JPG | 4.1782 | 3.8945 | 0.6847 | 0.7089 | coarse_em5_nm1__yaw_p0p5 | -5.000 | -1.000 | 0.500 | True | False |
| 0002.JPG | 4.7278 | 4.3484 | 0.4725 | 0.5537 | coarse_em3_nm5__yaw_m1 | -3.000 | -5.000 | -1.000 | True | False |
| 0003.JPG | 4.3905 | 3.8838 | 0.6980 | 0.7353 | coarse_ep0_np5__yaw_m1 | 0.000 | 5.000 | -1.000 | True | False |
| 0004.JPG | 2.9589 | 2.6933 | 0.8933 | 0.9670 | coarse_ep5_np5__yaw_p1 | 5.000 | 5.000 | 1.000 | True | False |

## Batch Statistics
| Metric | Value |
| --- | ---: |
| num_images_processed | 5 |
| non_initial_accept_rate | 1.0 |
| regression_rate | 0.0 |
| mean_chamfer_improvement | 0.6961703300476074 |
| median_chamfer_improvement | 0.37941932678222656 |
| mean_overlap_improvement | 0.08348785952858781 |
| feature_top1_safe_rate | 0.2 |
| feature_top5_contains_visual_top1_rate | 0.2 |

## Offset Distribution
| Statistic | East | North | Yaw | Alt |
| --- | ---: | ---: | ---: | ---: |
| mean | -1.2 | 0.8 | -0.1 | 0.0 |
| median | -3.0 | 0.0 | 0.0 | 0.0 |
| std | 3.487119154832539 | 3.815756805667783 | 0.8 | 0.0 |
| most common | -3.0 | 0.0 | - | - |

## Feature Scorer Analysis
| Scorer | Top1 safe rate | Top5 contains visual top1 rate | Mean Spearman vs chamfer | Mean Spearman vs overlap |
| --- | ---: | ---: | ---: | ---: |
| unweighted | 0.2 | 0.2 | 0.0739095755845682 | -0.06927500944544729 |
| uniform | 1.0 | 0.2 | 0.0739095755845682 | -0.06927500944544729 |
| dom_edge | 1.0 | 0.2 | 0.0540909281726208 | -0.06562696780151966 |
| depth_gradient | 1.0 | 0.0 | 0.017631501616220978 | -0.10860165400277064 |
| combined | 1.0 | 0.0 | 0.03841988161706056 | -0.07361151924772262 |
| low_texture_downweight | 1.0 | 0.0 | 0.004789891272406743 | -0.10463876411569625 |

## Interpretation
- does batch local search generalize: True
- regression rate is zero: True
- systematic offset hypothesis: most common selected east/north offset (-3.0, 0.0) count 1/5
- feature scorer generalizes: False
- any structure weight helps: False
- recommended next step: P13 should implement coarse-to-fine candidate refinement with visual-safe selection.

## Conclusion
Case B: Batch search safely improves images with dispersed offsets; this supports per-image local visual-safe refinement.
