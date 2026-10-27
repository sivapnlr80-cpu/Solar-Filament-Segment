# Solar Filament Segmentation Challenge 2026 - solution

Multi-view Mask R-CNN with a crop refiner for panoptic solar-filament segmentation.
Public leaderboard PQ **0.39**; held-out cross-validation PQ **0.442 +/- 0.014** (3 folds, official per-annotator metric).

Repository: <https://github.com/sivapnlr80-cpu/Solar-Filament-Segment> (MIT license). Report: `report.pdf` (4 pages). Competition: <https://www.kaggle.com/competitions/filament-segmentation-2026>

## Reproduce the submitted predictions

```bash
pip install -r requirements.txt            # Python 3.13 environment of the Kaggle run
jupyter notebook notebooks/pipeline.ipynb   # runs the whole pipeline and writes submission.csv
```

The notebook needs (1) the competition's `test_images/` folder under `./data` (or `/kaggle/input`) and
(2) two checkpoints that are **stored in this repository** under `weights/`: `maskrcnn_allcat_fold0.pth` (Mask R-CNN) and
`refiner.pth` (crop U-Net). GitHub rejects files over 100 MB, so each checkpoint is kept as parts of at most 50 MB
(`*.pth.partNN`); `src/filament/weights.py` concatenates them and verifies every part and the final file by SHA-256
(`weights/MANIFEST.json`). Nothing is downloaded and nothing is re-trained. It takes about 10 minutes on one T4 GPU
for the 180 test images. No other file is required. (The same two files are also attached to release `v1.0`.)

Reproduction check (Kaggle T4, same checkpoints): the code in `src/filament/` reproduces the submitted
`submission.csv` - 1294 of 1296 rows are byte-identical (174 of 176 non-empty images identical; the two differing
rows match the submitted masks with IoU > 0.95).

## Pipeline

| step | what | where |
|---|---|---|
| 1 | resize 2048 -> 1024, 4 test-time views (orig / h-flip / v-flip / rot180) | `src/filament/infer.py` |
| 2 | Mask R-CNN ResNet-50-FPN v2 (all annotation categories), detection score >= 0.7 | `src/filament/models.py` |
| 3 | native-resolution crop (1.8x context) -> 256x256 U-Net (ResNet-18 encoder) refiner, threshold 0.5 | `src/filament/models.py`, `infer.py` |
| 4 | cluster masks across views (IoU > 0.5), keep clusters present in >= 3 of 4 views, majority-vote mask | `infer.py` |
| 5 | score-ordered panoptic painting (no overlaps), min. area 200 px, RLE counts | `infer.py` |

## Repository layout

```
src/filament/      inference package (models, pipeline, official-PQ metric)
notebooks/         pipeline.ipynb - end-to-end pipeline illustrated and run
training/          train_detector.py (Mask R-CNN), train_refiner.py (crop U-Net)
experiments/       the scripts behind every number in the report (Kaggle-kernel scripts)
results/           CSV / log outputs of those scripts
tools/             make_notebook.py (regenerates the notebook)
tests/             synthetic-data unit tests: python -m unittest discover -s tests
```

## Training

* `training/train_detector.py` - 5-fold split over image files (`KFold`, shuffle, seed 42; identical in every script);
  the submitted model is fold 0 (`CFG['folds_to_train'] = [0]`), trained on all four annotation categories collapsed
  into one filament class; AdamW 1e-4, batch 2 x 4 accumulation, 10 epochs, flips.
* `training/train_refiner.py` - crop U-Net, BCE + Dice, AdamW 3e-4, 15 epochs.

Both read the competition data from `/kaggle/input` or `./data` and write the checkpoints to `/kaggle/working`.

## Experiments reported in the paper

| script | question | result file |
|---|---|---|
| `experiments/v50_allcat_vs_old_and_fragment_merging.py` | all-category vs categories 1-2 detector; fragment merging | `results/oof/allcat_vs_old.csv`, `fragment_merge.csv` |
| `experiments/v55_threshold_sweep_fold0.py` | score threshold x view agreement, fold 0 | `results/oof/thr_sweep.csv` |
| `experiments/v57_threshold_confirm_folds12.py` | the same on folds 1 and 2 (needs the older categories-1-2 checkpoints) | `results/oof/folds12_confirm.csv` |
| `experiments/v52_*`, `v53_*` | oversampling images with faint filaments | `results/ablations/faint_*` |
| `experiments/v54_boundary_postprocessing.py` | erosion / dilation / closing, refiner threshold | `results/ablations/boundary_*` |
| `experiments/v60_gbdt_candidate_rescoring.py` | learned (LightGBM) candidate rescoring | `results/ablations/*rescoring*` |
| `experiments/v64_consensus_soft_target_refiner_*.py` | refiner on consensus soft targets (x2 baseline restarts, all-annotator hard targets) | `results/ablations/consensus_refiner_*.csv` |

Held-out numbers use our re-implementation of the organizers' metric (PQ per annotator-image entry against all
predictions of the image; `src/filament/metrics.py`). The refiner is re-trained on the other folds for every held-out
fold, so it never sees validation images.

## Data and rules

Only the competition's training images / annotations are used for training, and only the provided `test_images`
at inference. No external data and no public MAGFiLO ground truth of the test images.
