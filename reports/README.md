# Reports

Figures produced while exploring and validating the dataset. They describe the
**data**, not any particular training run — per-run figures (training curves,
confusion matrices, ROC curves) are written to that run's results directory
instead, and are gitignored.

## Figures

| File | What it shows |
|---|---|
| `figures/01_split_distribution.png` | Image counts across the train / validation / test splits |
| `figures/02_class_distribution.png` | Images per severity class |
| `figures/03_class_proportions_heatmap.png` | Class proportions within each split, confirming the stratified split held |
| `figures/04_class_imbalance_analysis.png` | Imbalance ratios and severity score |
| `figures/05_sample_images.png` | Sample fundus images per class |
| `figures/06_image_dimensions.png` | Image dimensions before resizing |
| `figures/augmentation_detection_report.png` | Check for pre-augmented images in the source export |
| `figures/dataset_quality_analysis.png` | Corrupt, duplicate and off-size image checks |

## Regenerating the underlying numbers

The statistics behind the imbalance and quality figures come from
`dr.data.analysis`:

```python
from dr.data import analysis

report = analysis.imbalance_report("data/processed")
print(report["imbalance_ratio"], report["severity_level"])

verification = analysis.verify_labels("data/processed")
print(verification["issues"])
```

`scripts/preprocess.py` prints both as part of its run.
