# Data Format


## 1. Clinical tabular data

Recommended format: a single CSV (UTF-8), one row per subject.

- `ID` — unique subject identifier (string). Used to fix the sparsity-stress cohort across
  all mask ratios and across both the mHC and RAG-corrected pipelines.
- Demographic/clinical feature columns (numeric or categorical, encoded numerically).
- Label columns (see below).

### Primary diagnosis labels

| Label | Meaning |
|---|---|
| `NC`  | Normal cognition |
| `MCI` | Mild cognitive impairment |
| `DE`  | Dementia |

### Downstream labels

- `etiology` — e.g., AD, LBD, VD, FTD (adapt to your label set).
- `subtype` — e.g., MCI subtypes (MCI_A, MCI_AM, MCI_Na, MCI_NaM).

### Missing-value handling (important)

Raw `NaN` values must **not** be fed directly into the MLP. For each missing position:

1. **Impute** the value — either numerical imputation or replacement with random noise.
2. **Record a binary missingness mask** alongside the feature vector:
   - `observed = 0`
   - `missing / masked = 1`

The model consumes the imputed features together with the mask, so it can distinguish real
observations from imputed/masked ones. This same masking mechanism drives the feature-
sparsity experiment (see [REPRODUCIBILITY.md](REPRODUCIBILITY.md)).

## 2. MRI data

- Format: NIfTI volumes — `.nii` or `.nii.gz`.
- Referenced by path from a manifest/JSON (image path, optional label path, modality), e.g.
  the structure used by `adRAG/UniBrain-master/Brain_MRI/models/utils_file/seg.json`
  (paths shown there are placeholders — replace with your own).
- A path field in the clinical/manifest data links each subject `ID` to its MRI volume(s).

## 3. Example CSV header

```
ID,age,sex,mmse,cdr,<clinical_feature_1>,...,<clinical_feature_k>,mri_path,NC,MCI,DE,etiology,subtype
```

`mri_path` points to the subject's NIfTI file under your `mri_root`.
