# L515 Demo Positive Experiment Report

Status: `blocked`

## Raw Data

- Raw dir: `/home/hjc/coSpace/DLA_Final/vision/data/l515_demo_raw`
- Accept images: `0`
- Reject images: `0`

## Class Counts

| class | label | count |
| --- | --- | ---: |
| `flexible_wrapper` | accept | 0 |
| `dirty_wrapper` | accept | 0 |
| `tissue_napkin` | accept | 0 |
| `cigarette_butt` | accept | 0 |
| `garbage_bag` | accept | 0 |
| `small_misc` | accept | 0 |
| `rigid_plastic_bottle` | reject | 0 |
| `drink_can` | reject | 0 |
| `bottle_cap` | reject | 0 |
| `paper_cardboard` | reject | 0 |
| `glass_metal` | reject | 0 |

## Result

- Block reason: raw dataset needs at least one accept and one reject image
- Minimum required: accept `300`, reject `300`
- Training was skipped by design.

## Notes

- This experiment does not modify the v0.3 queue contract.
- This experiment does not replace the recommended export or AGX default model.
- Soft plastic wrappers, candy wrappers, snack bags, dirty film, and mixed-material wrappers are labeled accept for this experiment.
