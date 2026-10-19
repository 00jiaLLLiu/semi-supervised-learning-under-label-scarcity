# Semi-Supervised Learning under Label Scarcity

Course project (CSI 5155, University of Ottawa, Dec 2024) comparing supervised and semi-supervised learning on the mushroom-consumption split of the [UCI Drug Consumption Quantified](https://archive.ics.uci.edu/dataset/373/drug+consumption+quantified) dataset, under controlled labeled-data scarcity.

## Overview

Labels are expensive — this project asks how far unlabeled data can take you. A supervised **Gradient Boosting** baseline (hyperparameters tuned with `GridSearchCV`) is compared against four semi-supervised algorithms across labeled fractions of **10%–50%**:

- **Self-Training** (`sklearn.semi_supervised`) — iterative pseudo-labeling
- **Co-Training** — two classifiers trained on random feature-view splits, labeling for each other
- **SemiBoost** ([LAMDA-SSL](https://github.com/LAMDA-SSL/LAMDA-SSL)) — boosting over pseudo-labels (includes a patch for a `np.float` deprecation in the library)
- **Label Spreading** (`sklearn.semi_supervised`) — graph-based label propagation

Models are compared on ROC-AUC, precision, recall, F1-score, and accuracy, with AUC/F1 line plots and heatmaps across labeled fractions.

## Key Findings

- **Self-Training** — AUC climbs with labeled fraction and edges past the baseline at 50% (0.8157 vs. 0.8117); at 10% its F1 is marginally higher. But the model teaches itself, so it can only reinforce what it already believes — early mistakes get baked in deeper each round, which is why it never truly beats the baseline. The real win: same performance with only half the labels.
- **Co-Training** — never exceeds AUC 0.8 or F1 0.55 at any fraction. Co-training only works when each view can solve the task on its own (the textbook example: a web page's text vs. its links). Here the views were random halves of the same 12 features — two half-blind models copying each other's homework, so errors compounded instead of cancelling out.
- **SemiBoost** — F1 holds at 0.6681 (above baseline) across all fractions while AUC stays near baseline (exceeding it at 40%). Boosting re-weights the points it gets wrong each round, so noisy pseudo-labels get outvoted by consistent signal over time. That's why its F1 barely moves as labels shrink — of the four methods, it needs labels the least.
- **Label Spreading** — best AUC at every fraction (0.8142–0.8158, all above baseline), yet F1 falls as labeled fraction grows (0.6357 → 0.5856). It spreads each label to nearby points, so one labeled example per cluster is enough — hence the flat, high AUC. But "nearby" favors the majority class: more labels just spread more majority votes, drowning out the minority, which is why F1 falls as labels increase.

Full analysis with figures is in `report/Summary.pdf`.

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook semi_supervised_learning.ipynb
```

Run cells top to bottom — the baseline training cell saves `model/baseline_model.pkl`, which a later evaluation cell reloads. The dataset is included in `data/`.

## Repository Structure

```
├── semi_supervised_learning.ipynb  # baseline + 4 semi-supervised methods, full comparison
├── data/
│   └── mushroom_dataset.csv        # derived mushroom-consumption dataset
└── report/
    └── Summary.pdf                 # written analysis with figures
```
