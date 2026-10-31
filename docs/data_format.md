# Input and output format

All identifiers in an analysis folder must be pseudonymous and consistent. Source and target identifiers must be globally distinct. Paths within input tables are relative to their containing folder. `.npz` inputs must contain numeric or Unicode arrays, never pickled Python objects.

## Feature-level screening data

Each cohort has a separate folder containing:

* `patients.csv`: one row per patient, with `patient_id`, observed binary AS `label`, `age`, `sex_female` (0/1), `sbp`, `dbp` and, for MIMIC or source-compatible external evaluation, `hr`. Unknown clinical values are empty/NaN. AF is not a predictor. Optional `duplicate_group` joins patients with duplicate waveforms into an indivisible validation group. Optional `pre_tavi` records independently curated case membership (1), known absence (0) or unknown (empty). It is used only by `stratify`.
* `segments.csv`: unique `segment_id`, `patient_id` and `stage=index`, in exactly the feature-array order. All included patients need at least one jointly usable index recording. Longitudinal rows belong in a different dataset.
* `features.npz`: `segment_id` and `patient_id` Unicode arrays, `pyppg` (segments × 918), `embedding` (segments × 512), and optional `encoder_input` (segments × 1250). Morphology missing values use NaN; embeddings must be finite. `encoder_input` is required by the source-trained Net1D.

Morphology columns follow `configs/feature_schema.json`. That file also supplies the domain mapping and the separate 510-feature inference family. The extraction schema is not a patient file.

EDU internal classifiers use patient-level medians. MIMIC PPG classifiers fit segments with weights summing to one per patient, and prediction scores are averaged within patients before calibration. The clinical-only classifier uses one row per patient. MIMIC-trained models evaluated in EDU retain source-compatible segment processing and the MIMIC clinical feature order; do not substitute the EDU internal feature matrix without checking its provenance.

## Raw signal input

For EDU, provide `patients.csv`, `segments.csv` and individual `.npy` waveforms. The segment table additionally has `signal_path`, `fs=60`, `cohort=EDU` and `processing_state`. `processing_state=raw` applies the original NeuroKit cleaner once; `neurokit_cleaned` uses the supplied cleaned samples. Every segment has 900 finite samples. Keep the raw trace available when applying raw-signal quality checks. If the source has already been quality screened, use `extract` directly with its true processing state instead of applying a second cleaner.

For MIMIC preparation, `parents.csv` has `patient_id`, unique `parent_id`, `signal_path`, and optional `start_time`. Each `.npy` parent contains 3,750 native 125-Hz physical PLETH samples. The inventory must already resolve overlapping physical/time intervals and repeated sources. The compact adapter does not download WFDB records or resolve clinical identities. Parents are ordered by `start_time,parent_id` when timestamps exist, then selected with the original endpoint/midpoint traversal and seeded half order. Each patient contributes at most 20 saved 15-second native-rate segments. All three 10-second subwindows of a parent must pass upstream Ext-PPG SQI. Saved values are raw physical samples; QC-filtered values are not exported.

Extraction records failures in `extraction_flow.csv`, marks patient inclusion in `patient_flow.csv`, and writes only jointly usable recordings into the output feature dataset. These are technical inclusion records, not model-based patient selection. Clinical labels and missingness remain separate from signal quality.

## Frozen validation folds and transfer roles

Study EDU folds must be supplied as JSON:

```json
{
  "patient_id": ["SYN_A", "SYN_B"],
  "folds": [
    {"fold": 1, "train": [0], "test": [1], "inner": []}
  ]
}
```

This is a structural illustration, not a runnable two-patient experiment. Real folds need both classes, complete outer-test coverage and complete inner-OOF coverage. Indices refer to the listed patient order and the matching `patients.csv`. Original EDU folds inherited joint-endpoint strata; they cannot be regenerated exactly from AS labels alone. Retaining those fold assignments does not add AR as a modelling task. For new illustrative data use the demo configuration or supply explicitly reviewed folds.

Transfer `--roles` accepts either a role dictionary or a dictionary under `splits`. Role keys are `A_train`, `A_val`, `A_internal_test`, `B_adapt`, `B_train`, `B_val`, `B_test`, each containing patient IDs. `A_*` refers to MIMIC; `B_*` refers to EDU. `B_train ∪ B_val = B_adapt`; all test patients are excluded from fitting, adaptation, calibration and threshold selection within this experiment. When roles are not supplied the recorded AS source split and whole-EDU 20/80 rule are generated with seed 61007. Exact study reproduction uses the frozen roles, not a different patient ordering or cohort.

## Morphology inputs

Cross-sectional statistics use the feature-level dataset above and median-aggregate MIMIC segments by patient. Longitudinal CSV contains one row per `patient_id,stage`, where `stage` is `pre`, `post` or `followup`, and all 510 inference-feature columns. Missing values stay NaN. Both paired and unpaired pre/post observations enter the random-intercept model. Follow-up tests use the available patient pairs separately.

Optional waveform-template inputs use `segments.csv` with `patient_id`, `stage`, `cohort`, `fs`, `signal_path` and `processing_state`. Template generation uses complete onset-to-next-onset pulses. Real-time templates do not time-warp beats; all derivative channels share the PPG amplitude scale. Minimum within-segment and within-patient support is 50%; population display coverage is 90%. Population summaries use one template per patient. The CLI additionally exports paired pre/post summaries using matched patient IDs and shared bootstrap draws. All exported groups use the common contiguous interval meeting 90% patient coverage. The function API also accepts shared bootstrap indices explicitly.

## Outputs

* Validation: `folds.json`, each model's `oof.csv`, fold predictions, local model bundles, selected-feature audit and performance table.
* Transfer: frozen `roles.json`, each model's `zero_shot.csv` and `adapted.csv` on identical patients, local model bundles and comparison metrics.
* Evaluation: patient-bootstrap confidence intervals, reliability groups and optional subgroup tables.
* Stratification: selected count, case count, case capture, selected-group proportion and enrichment. Counts round up and include boundary score ties. AS and pre-TAVI use separate case totals.
* Morphology: complete-family cross-sectional tests, pre/post mixed-model results, paired follow-up tests, and optional real-time templates.

These output files are excluded from Git by default. Only the synthetic-data generator is distributed.
