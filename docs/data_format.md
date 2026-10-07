# Data formats

## Images for split generation

```text
<image_directory>/
    <case_id>_0000.nii.gz
```

The filename prefix becomes the case ID. The output JSON contains one object per
fold, with `train` and `val` lists of case IDs.

## Location probabilities

| Input column | Output column |
| --- | --- |
| `id`, or the selected identifier column | `New_ID` |
| `Prob_Basal` | `p_basal` |
| `Prob_Brainstem` | `p_brainstem` |
| `Prob_Cerebellum` | `p_cerebellum` |
| `Prob_Lobar` | `p_lobar` |
| `Prob_Thalamus` | `p_thalamus` |

Identifiers must be nonempty and unique. They are read as strings to preserve
leading zeros. All five probability columns are required. Values must be finite
and between zero and one, with row sums within `1e-6` of one. Additional columns
are not copied to the output.

## CT and mask directories

```text
data/
|-- images/
|   |-- imagesTr/<case_id>_0000.nii.gz
|   `-- imagesTs/<case_id>_0000.nii.gz
|-- masks/
|   |-- labelsTr/<case_id>.nii.gz
|   `-- labelsTs/<case_id>.nii.gz
|-- clinical/
|   |-- clinical_info_train_v2.csv
|   `-- clinical_info_test_v2.csv
`-- splits_final.json
```

CT and mask arrays must share the same grid. Masks use positive values for the
hematoma. Clinical and feature tables use `New_ID` as the join column;
column mappings are defined in `features/config.py`.

## Location feature tasks

The task file is a JSON list. Each item contains `img` (CT path), `msk` (mask
path), `id` (case identifier), `label` (integer class) and `split` (`train` or
`test`). Use absolute paths or paths relative to the repository root.

The feature table contains `id`, `subset`, `label` and numeric feature columns.
Class order is basal ganglia, brainstem, cerebellum, lobar and thalamus, encoded
as integers zero through four. Classification requires all five classes in
training and enough samples per class for the selected fold count.
