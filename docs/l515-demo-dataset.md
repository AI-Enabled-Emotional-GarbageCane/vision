# L515 Demo Dataset

This dataset is for the AGX + Intel RealSense L515 demo-angle correction.
It is not part of the public v0.3 queue contract and does not replace the
recommended ONNX export by itself.

## Raw Folder Layout

Place RGB snapshots copied from the AGX under `data/l515_demo_raw/`.

Accept classes:

- `flexible_wrapper`: candy wrappers, snack bags, soft plastic film, mixed-material wrappers
- `dirty_wrapper`: food-stained packaging and contaminated film
- `tissue_napkin`: tissues, napkins, paper towels
- `cigarette_butt`: cigarette butts
- `garbage_bag`: small trash bags and bag fragments
- `small_misc`: small non-recyclable miscellaneous trash

Reject classes:

- `rigid_plastic_bottle`: plastic bottles and rigid recyclable plastic containers
- `drink_can`: aluminum or metal drink cans
- `bottle_cap`: plastic or metal bottle caps
- `paper_cardboard`: clean paper and cardboard
- `glass_metal`: glass and metal recyclables

Policy for this experiment: flexible plastic wrappers, candy wrappers, snack
bags, dirty film, and mixed-material wrappers are `accept`.

## Run Command

Initialize the raw folders and write the current blocked/ready report:

```sh
python3 scripts/run-l515-demo-experiment.py --init-only
```

After at least 300 accept images and 300 reject images are present, run:

```sh
python3 scripts/run-l515-demo-experiment.py
```

The script builds `data/training/l515_demo_yolo_cls`, merges the combined
training dataset, trains two serial-numbered candidates under
`runs/l515-demo-positive/`, evaluates all required weak-label sets, writes
threshold sweeps, and creates `runs/l515-demo-positive/l515_demo_fix_report.md`.

If the raw dataset is below the minimum gate, training is skipped by design.
