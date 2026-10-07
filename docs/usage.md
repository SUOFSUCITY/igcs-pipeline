# Usage

Run commands from the repository root. Replace angle-bracket placeholders with
your paths and settings.

## Parameters

`PIPELINE_PARAMETERS` points to a JSON object with the keys listed in
`parameters.example.json`. Each stage reads only the keys it uses.
Set these values in your local copy:

| Group | Values |
| --- | --- |
| `segmentation` | Spacing (three positive numbers), HU bounds, crop margin, model arguments, loss weights and smoothing, optimizer settings, gradient clipping, decay and EMA settings |
| `localization` | Brain HU bounds, closing iterations, ring width in mm, tree count, seed and fold count |
| `imaging` | Intracranial HU threshold |
| `pash` | Low/high percentiles, minimum mask size and low-density voxel count |

`segmentation.model` is an object passed to `create_efficient_mednext`.
It accepts `num_input_channels`, `num_classes`, `model_id`, `n_channels`,
`kernel_sizes`, `strides`, `uniform_dec_channels`, `deep_supervision` and `mode`.
Use model dimensions compatible with your data and checkpoint.
Loss weights form a numeric list; spacing follows the image axis order.

For PowerShell:

```powershell
$env:PIPELINE_PARAMETERS = '<parameter_json>'
$env:PIPELINE_DATA_DIR = '<data_directory>'
$env:PIPELINE_FEATURES_DIR = '<feature_output_directory>'
$env:EFFICIENTMEDNEXT_REPO = '<upstream_source_directory>'
```

The default data directory is `data/`; feature tables go to `outputs/features/`.

## Segmentation

```sh
python scripts/generate_splits.py --images-dir <imagesTr> --output <splits_final.json> --num-folds <fold_count> --seed <seed>
python segmentation/preprocess.py --data_dir <data_directory> --output_dir <preprocessed_directory>
python segmentation/train.py --data_dir <data_directory> --output_dir <checkpoint_directory> --epochs <epochs> --batch_size <batch_size> --lr <learning_rate> --patch_size <x> <y> <z> --num_iterations_per_epoch <iterations> --seed <seed>
python segmentation/finetune.py --data_dir <data_directory> --pretrained_weights <checkpoint> --output_dir <checkpoint_directory> --epochs <epochs> --warmup_epochs <warmup> --batch_size <batch_size> --lr <learning_rate> --patch_size <x> <y> <z> --num_iterations_per_epoch <iterations> --seed <seed>
```

Add `--preprocessed_dir` to train on cached arrays. Select a fold with `--fold`.
Training uses CUDA and supports distributed launching with PyTorch.

## Location classification

```sh
python localization/features.py --tasks <task_json> --output <feature_csv>
python localization/classifier.py --data <feature_csv> --output-dir <new_output_directory>
python features/export_location.py --input <probability_csv> --output <location_encoding_csv> --id-column id
```

The classifier writes cross-validation predictions, test predictions, metrics and
fitted model artifacts. Probability export validates identifiers and probabilities.
Use new output paths for these commands.

## Imaging features

```sh
python features/imaging.py
python features/pash.py
```

These commands process the training and test directories configured in
`features/config.py`. Existing case IDs in the output tables are skipped.
Use a new feature output directory when changing parameter values.

`features/legacy_matrix.py` merges clinical, location, imaging and PASH tables
for the conservative-treatment cohort (`Surgery == 0`), fills missing feature
values with each table's median, and adds interaction and indicator columns.
