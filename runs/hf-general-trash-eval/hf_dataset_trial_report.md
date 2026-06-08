# Hugging Face General Trash Dataset Trial

Date: 2026-06-04

## Sources Tried

1. `mnemoraorg/256x256-litter-sort-annotated-wastes`
   - Link: https://hf.co/datasets/mnemoraorg/256x256-litter-sort-annotated-wastes
   - Local source: `data/sources/hf_mnemora_256_litter_sort`
   - YOLO cls dataset: `data/training/hf_mnemora_litter_yolo_cls`
   - Mapping: `trash -> accept`; `cardboard`, `paper`, `metal`, `glass -> reject`; `plastic` ignored.
   - Materialized total: 2044 rows after train reject balancing.
   - Note: HF card text claims a larger balanced dataset, but the downloadable repo file tree/materialized local content did not match that full count.

2. `1ease2/waste-garbage-management-dataset`
   - Link: https://hf.co/datasets/1ease2/waste-garbage-management-dataset
   - Local source sample: `data/sources/hf_1ease2_waste_sample`
   - YOLO cls dataset: `data/training/hf_1ease2_waste_sample_yolo_cls`
   - Mapping: `trash -> accept`; `battery`, `cardboard`, `glass`, `metal`, `paper`, `plastic -> reject`.
   - Downloaded selected sample: 250 per mapped class, 1750 raw images.
   - Materialized total: 1050 rows after train reject balancing.

3. `omasteam/waste-garbage-management-dataset`
   - Link: https://hf.co/datasets/omasteam/waste-garbage-management-dataset
   - Status: duplicate source candidate.
   - File-set check: exactly the same 19764 repo paths as `1ease2/waste-garbage-management-dataset`.
   - Decision: not merged into training/evaluation dataset to avoid duplicate weighting.

## Combined HF Dataset

- Path: `data/training/hf_general_trash_sample_yolo_cls`
- Sources merged: mnemora + 1ease2 sample
- Total rows: 3094
- Split counts:
  - train/accept: 271
  - train/reject: 542
  - val/accept: 59
  - val/reject: 1083
  - test/accept: 57
  - test/reject: 1082

## Weak-Label Evaluation

Evaluated on the combined HF test split with the default accept gate threshold `0.76`.

| model | weak agreement | gate accept recall | reject false accept |
| --- | ---: | ---: | ---: |
| `realwaste-accuracy-002` | 93.42% | 1.75% | 1.48% |
| `general-trash-positive-002` | 72.43% | 5.26% | 2.96% |

Threshold sweep showed that lowering the gate is not enough for the safety baseline:

- `realwaste-accuracy-002` at threshold `0.50`: accept recall 7.02%, reject false accept 2.03%.
- `general-trash-positive-002` at threshold `0.50`: accept recall 42.11%, reject false accept 25.97%.

## Interpretation

These HF sources are useful as additional weak-label coverage, but the `trash` classes do not look like the current models' learned `accept` domain. The safety baseline rejects almost all HF `trash`; the general-trash-positive model accepts more raw positives, but lowering the gate quickly damages reject safety.

This dataset should be used for a controlled fine-tune candidate, not as a direct threshold-only fix. `omasteam` should stay excluded unless a later audit finds content differences from `1ease2`.
