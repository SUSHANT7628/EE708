# ShiftGuard10: Robust Image Classification under Class Imbalance and Distribution Shift

EE708 course project (IIT Kanpur, Prof. Rajesh M. Hegde), Group 8: Aarnav Gupta, Aarush Singh,
Kaushey, Prince Pathak, Sushant Kumar.

A **WideResNet-40-10 ensemble** for 10-class classification of 32×32 images. The training set is
**heavily imbalanced (50:1)**, and the test images carry a **distribution shift**, including grey
cutout patches. We optimised for **Macro F1**. Best validation **Macro F1 = 0.9206**.

## Data

`train_labels.csv` lists 29,400 labelled training images. The counts per class show the 50:1
imbalance:

| airplane | automobile | bird | cat | deer | dog | frog | horse | ship | truck |
|---|---|---|---|---|---|---|---|---|---|
| 5,000 | 5,000 | 5,000 | 5,000 | 4,000 | 4,000 | 500 | 500 | 300 | 100 |

## Method (all in `Group_8.ipynb`)

| Problem | What we used |
| --- | --- |
| Backbone | `WideResNet(depth=40, width=10, drop=0.3)`, about 55.8M parameters |
| Class imbalance | **BalancedSoftmaxLoss**: log class-frequency added to the logits, with label smoothing 0.1. **Weighted random sampling** uses square-root inverse-frequency weights, which are deliberately softer than 1/n so they don't double-correct together with Balanced Softmax |
| Distribution shift | **AugMix** (severity 2) in place of AutoAugment, **MixUp** and **CutMix**, random erasing, Gaussian noise, and a custom **GreyCutout**. GreyCutout fills a random 5–12 px square with normalised grey, half the time, mimicking the grey cutout artefacts in the test images |
| Optimisation | SGD with Nesterov momentum, cosine LR schedule, 300 epochs, batch 256 |
| Mixed precision | `torch.autocast` and `GradScaler` (AMP) on a Tesla T4 |
| Flatter minima | **SWA** (stochastic weight averaging) over the last 75 epochs (from epoch 225), with BN statistics recomputed |
| Validation | 95/5 stratified train/validation split. Macro F1 on the validation set after every epoch |
| Test-time robustness | **TTA**: 30 augmented passes plus one clean pass weighted 3×, probabilities averaged |
| Ensemble | 3 seeds (23, 71, 314), with probabilities averaged across models |

## Results

| Seed | Validation Macro F1 |
| --- | --- |
| 23 | **0.9206** |
| 71 | 0.8931 |
| 314 | 0.9146 |
| Mean | 0.9094 |

The final Kaggle submission is the 3-seed ensemble with 30-round TTA per model. The notebook
outputs show the per-epoch training log, the SWA phase, the classification report for the best
seed, and how the submission was assembled.

## Files

| File | Contents |
| --- | --- |
| `Group_8.ipynb` | Full pipeline: data loading, augmentations, model, training with SWA/AMP, TTA, ensemble, submission (with outputs) |
| `tipynb` | Kaggle data-download and setup notebook (JSON notebook without the `.ipynb` extension) |
| `Group8_report.pdf` | Project report |
| `Group_8.pptx` | Presentation |
| `train_labels.csv`, `classes.txt`, `sample_submission.csv` | Competition label and submission files (images are not included) |

## Run

Open `Group_8.ipynb` on Kaggle or Colab with a GPU and attach the competition data. One seed takes
about 3 hours on a T4 (300 epochs), and the ensemble trains 3 seeds.

## Key learnings

- Layered robustness (imbalance-aware loss, shift-targeted augmentation, SWA, TTA, ensembling) beat any single technique.
- Balanced Softmax plus a full inverse-frequency sampler over-corrects towards minority classes. Square-root weights were the better balance.
- An augmentation built from the actual test artefact (GreyCutout) is cheap and targeted.
