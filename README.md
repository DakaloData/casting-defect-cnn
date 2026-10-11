# Casting Defect Detection with Convolutional Neural Networks

A computer vision project that classifies photos of cast metal parts (submersible pump impellers) as **defective** or **acceptable**, the kind of check a production line does by eye today. The notebook builds a baseline CNN, tries to improve on it, and compares seven configurations on the same untouched test set.

**Business goal:** miss as few defective parts as possible (false negatives) without flooding the line with unnecessary re-checks (false positives).

## Results

Evaluated on a held-out test set of 715 images (453 defective, 262 acceptable). Images are 96 x 96 greyscale.

| Model | Threshold | Accuracy | Precision | Recall | Missed defects | False alarms |
| --- | --- | --- | --- | --- | --- | --- |
| **Baseline CNN** | 0.50 | **99.86%** | 100% | 99.78% | **1** | **0** |
| Improved v1 (rotation/flip augmentation) | 0.50 | 50.91% | 93.22% | 24.28% | 343 | 8 |
| Improved v1, tuned threshold | 0.11 | 63.50% | 63.45% | 100% | 0 | 261 |
| Improved v2 (revised augmentation) | 0.50 | 98.46% | 100% | 97.57% | 11 | 0 |
| Improved v2, tuned threshold | 0.41 | 98.74% | 98.47% | 99.56% | 2 | 7 |
| Tuned v3 (learning rate, dropout) | 0.50 | 95.94% | 100% | 93.60% | 29 | 0 |
| Tuned v3, tuned threshold | 0.05 | 99.30% | 100% | 98.90% | 5 | 0 |

Thresholds were chosen on the validation set only, using an F2 score that weights recall twice as much as precision. The test set was used once per model.

## What I learned

- **A simple baseline can already be near the ceiling.** The images come from a fixed rig, so a three-block CNN reached 99.86% accuracy with one missed defect. The "improved" models did not beat it.
- **Augmentation has to match reality.** My first improved model used rotation and flips, but every casting in this dataset is photographed the same way round. Its accuracy fell to about 51%. Removing the rotation and flips brought the test results back.
- **Check the curves, not just the final score.** Even after that change, the validation curves for v2 and v3 stayed erratic, and the strong test results came from an early epoch whose weights early stopping restored. The instability is not fully resolved. Batch Normalisation behaving differently in training and evaluation is a likely cause that I did not isolate.
- **Threshold tuning trades one error for another.** It cannot fix a model whose probability ranking is poor, as v1 shows.
- **A single training run can mislead.** v2 and v3 use the same architecture and the same learning rate and dropout, yet v3 missed 29 defects at the default threshold against v2's 11.

## What the notebook covers

1. Business problem and dataset, with a plain-language introduction to CNNs
2. Exploratory analysis: class balance and sample images
3. Stratified train/validation split, with the test set untouched until evaluation
4. Baseline CNN from scratch
5. Improved model: batch normalisation, dropout, global average pooling, augmentation and class weighting
6. Threshold tuning, ROC and precision-recall curves, and a diagnosis of the training instability
7. A small hyperparameter search
8. Model comparison and reflection

## Data

The images are not included in this repository. Download **Real-life Industrial Dataset of Casting Product** from Kaggle (by Raviraj) and check its licence on the dataset page:
https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product

Place it next to the notebook like this:

```
data/
├── train/
│   ├── def_front/
│   └── ok_front/
└── test/
    ├── def_front/
    └── ok_front/
```

## Run it

1. Download this repository (green **Code** button, then **Download ZIP**) and unzip it.
2. Add the `data` folder as shown above.
3. Run `pip install -r requirements.txt`.
4. Run `jupyter notebook` and open `casting_defect_cnn.ipynb`.

Training runs on CPU and takes a while, about 80 seconds per epoch for the improved models. The notebook saves each trained model to a `checkpoints` folder and reloads it on later runs. Checkpoints are not included here, so the first run trains everything from scratch. The results above come from one run, so a fresh run may differ slightly.

## Next steps

- Try transfer learning with a pretrained network such as MobileNetV2 (not possible in the offline environment this was built in)
- Investigate the validation instability, for example Batch Normalisation settings or a lower learning rate
- Repeat the training with several random seeds and report the spread

## Tech stack

Python, TensorFlow/Keras (2.21), NumPy, pandas, scikit-learn, Matplotlib, Seaborn, Pillow
