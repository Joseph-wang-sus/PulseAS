# Input and output format

Provide analysis-ready records with consistent pseudonymous identifiers. Source and target patient IDs must be distinct. Paths in input tables are relative to their containing folder.

## Feature inputs

Each cohort has its own folder:

| File | Required contents |
| --- | --- |
| `patients.csv` | One row per patient: `patient_id`, observed binary AS `label`, `age`, `sex_female`, `sbp`, `dbp`; MIMIC/source-compatible transfer also uses `hr` |
| `segments.csv` | Unique `segment_id`, `patient_id`, `stage=index`, in feature-array order |
| `features.npz` | Unicode `segment_id` and `patient_id`; numeric `pyppg` (segments × 918) and `embedding` (segments × 512) |

Missing clinical/morphology values use empty cells or NaN. Embeddings must be finite. Unknown AS labels are not negatives. Optional `encoder_input` (segments × 1250) is required for Net1D. Arrays must not contain pickled objects. Feature names follow `configs/feature_schema.json`.

Optional patient fields: `duplicate_group` keeps duplicated recordings together in validation; `pre_tavi` supplies procedure-group membership for stratification (1 = case, 0 = known absence, empty = unknown). AF is not a predictor. EDU models use patient-level median features; MIMIC PPG models use patient-weighted segments.

## Signal inputs

Prepared 15-second recordings use a `segments.csv` containing `patient_id`, `segment_id`, `signal_path`, `fs`, `cohort`, `stage` and `processing_state`, alongside `patients.csv` and one-dimensional `.npy` waveforms. EDU uses 60 Hz; MIMIC uses native 125 Hz physical PLETH. EDU states are `raw` or `neurokit_cleaned`; MIMIC inputs use `raw`. Set the state accurately to avoid cleaning twice.

Use `extract` directly on prepared recordings. The optional `prepare` command performs automated signal checks. Its raw MIMIC adapter accepts `parents.csv` with `patient_id`, unique `parent_id`, `signal_path` and optional `start_time`, and native 30-second recordings; it requires separately supplied Ext-PPG utilities. Patient matching, manual screening and clinical annotation are completed before supplying these inputs.

## Validation assignments

`--folds` accepts JSON with the ordered `patient_id` list and a `folds` list. Each fold contains `fold`, patient-position indices `train` and `test`, and an `inner` list of `train`/`valid` indices. Provide the saved study assignments for exact replay. Every patient must have one outer-test assignment; no patient or duplicate group crosses partitions.

Transfer `--roles` accepts a dictionary (optionally under `splits`) of patient-ID lists: `A_train`, `A_val`, `A_internal_test`, `B_adapt`, `B_train`, `B_val`, `B_test`. A is MIMIC-III; B is EDU. `B_train` and `B_val` partition `B_adapt`. All test patients stay outside development. Without `--roles`, the configured source split and target 20/80 split are generated.

## Morphology and predictions

Longitudinal CSV inputs contain one row per `patient_id,stage` (`pre`, `post`, `followup`) and the 510 inference-feature columns. Optional waveform templates use the signal format above with the corresponding stage. Longitudinal data are separate from classifier inputs.

Prediction tables contain `patient_id`, binary `label` and calibrated `probability`. `stratify` optionally reads `pre_tavi` from the patient table. AS and pre-TAVI capture have separate denominators.

## Outputs

Validation and transfer write patient scores, model bundles and performance tables. Evaluation writes performance and clinical-comparison summaries; stratification writes case-capture and enrichment tables; morphology writes cross-sectional, pre/post and follow-up statistics and optional waveform summaries. Generated inputs, weights and outputs are excluded from Git.
