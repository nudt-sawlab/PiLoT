# P12 Batch Local Visual-Safe Candidate Search

## Purpose
P11 showed local visual-safe search improves one image. P12 applies the same fixed search policy to multiple EXIF-recovered query poses and checks generalization plus systematic offsets.

## Why Batch Is Necessary
Single-image search does not prove generalization. P12 keeps candidate generation fixed and does not tune the range for any one query image.

## Search Policy
- coarse east offsets: [0.0]
- coarse north offsets: [0.0]
- yaw refinement offsets: [0.0]
- topK visual for yaw: 0
- enable alt refine: False
- final pose policy: strict_visual_gate_then_best_chamfer

## Gate Policy
- strict visual gate: chamfer <= initial and overlap >= initial
- final selection: lowest chamfer among strict-pass candidates; fallback initial if none pass

## Batch Results
| Image | Initial chamfer | Selected chamfer | Initial overlap | Selected overlap | Selected candidate | East | North | Yaw | Strict non-initial? | Safe worse? |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |
| 0000.jpg | 9.6303 | 9.3503 | 0.5286 | 0.7123 | yaw_minus_1deg | 0.000 | 0.000 | -1.000 | True | False |

## Batch Statistics
| Metric | Value |
| --- | ---: |
| num_images_processed | 1 |
| non_initial_accept_rate | 1.0 |
| regression_rate | 0.0 |
| mean_chamfer_improvement | 0.27999210357666016 |
| median_chamfer_improvement | 0.27999210357666016 |
| mean_overlap_improvement | 0.18372555518727207 |
| feature_top1_safe_rate | 0.0 |
| feature_top5_contains_visual_top1_rate | 0.0 |

## Offset Distribution
| Statistic | East | North | Yaw | Alt |
| --- | ---: | ---: | ---: | ---: |
| mean | 0.0 | 0.0 | -1.0 | 0.0 |
| median | 0.0 | 0.0 | -1.0 | 0.0 |
| std | 0.0 | 0.0 | 0.0 | 0.0 |
| most common | 0.0 | 0.0 | - | - |

## Feature Scorer Analysis
| Scorer | Top1 safe rate | Top5 contains visual top1 rate | Mean Spearman vs chamfer | Mean Spearman vs overlap |
| --- | ---: | ---: | ---: | ---: |
| unweighted | 0.0 | 0.0 | 0.1333333333333333 | -0.5166666666666666 |
| uniform | 1.0 | None | 0.1333333333333333 | -0.5166666666666666 |

## Interpretation
- does batch local search generalize: False
- regression rate is zero: True
- systematic offset hypothesis: most common selected east/north offset (0.0, 0.0) count 1/1
- feature scorer generalizes: False
- any structure weight helps: False
- recommended next step: Inspect metric stability, query/pose quality, and search range before pipeline integration.

## Conclusion
Case C: Batch evidence is insufficient for generalization under the current fixed policy.
