# ICH imaging pipeline

Code for intracerebral hemorrhage CT segmentation, location classification and
imaging feature extraction.

## Structure

| Directory | Contents |
| --- | --- |
| `segmentation/` | Preprocessing, datasets, training, fine-tuning and losses |
| `localization/` | Location features, cascaded classification and evaluation |
| `features/` | Imaging and PASH features, probability export and table assembly |
| `scripts/` | Cross-validation split generation |
| `docs/` | Usage and data formats |

## Setup

Use Python 3.10 or later. Run commands from the repository root.

```sh
python -m pip install -r requirements.txt
```

Segmentation also requires `requirements-optional.txt` and the upstream
EfficientMedNeXt source. Set `EFFICIENTMEDNEXT_REPO` to that source directory.
Choose a PyTorch build compatible with your CUDA environment.

## Configuration and usage

Copy `parameters.example.json` to a local parameter file, fill the values needed
by your workflow, and set `PIPELINE_PARAMETERS` to its path.
Training settings such as the learning rate and patch size are command arguments.

See [usage](docs/usage.md) for each stage and
[data formats](docs/data_format.md) for image paths and table columns.

## Acknowledgements

See [third-party notices](THIRD_PARTY_NOTICES.md) for upstream references.
