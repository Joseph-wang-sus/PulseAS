# Analysis guide

## Main workflow

1. Prepare analysis-ready patient records and 15-second PPG recordings in the [input format](data_format.md).
2. Preprocess signals and extract waveform biomarkers and frozen AnyPPG representations.
3. Fit classifiers using patient-level nested validation separately in EDU and MIMIC-III.
4. Evaluate the MIMIC-III source model in EDU before and after local adaptation.
5. Summarise performance, clinical comparisons, AS/pre-TAVI score stratification and pulse morphology.

Patient matching, manual screening and clinical annotation precede the analysis. The public package starts from the resulting inputs and does not include operational procedures or historical study scripts.

## Features and models

Morphology extraction produces 918 summary features; statistical comparisons use the separate 510-feature family in `configs/feature_schema.json`. Frozen AnyPPG generates 512-dimensional representations from the central ten seconds. Top-10 morphology selection occurs within training folds.

| Identifier | Strategy |
| --- | --- |
| `full_pyppg` | Morphology + ExtraTrees |
| `clinical` | Clinical variables + L2 logistic regression |
| `anyppg` | Frozen AnyPPG + XGBoost |
| `fusion_top10` | AnyPPG + selected morphology + XGBoost |
| `neural_fusion` | AnyPPG + selected morphology + MLP |
| `clinical_fusion` | PulseAS–XGB |
| `clinical_neural_fusion` | PulseAS–MLP |
| `clinical_fusion_logistic` | PulseAS–L2 |
| `net1d` | Additional source-trained waveform model for transfer |

Use `--models` to choose a comma-separated subset where supported.

## Validation design

Study settings use five outer folds and three inner folds. All records from a patient stay together; preprocessing, feature selection and model selection use the corresponding training data. MIMIC-III patients contribute equal total training weight, and segment scores are aggregated before calibration.

EDU study runs require saved fold assignments through `--folds`. Transfer uses 20% of eligible EDU patients for adaptation and validation, with the same independent 80% test set before and after adaptation. Saved transfer assignments can be supplied through `--roles`.

Adaptation continues XGBoost trees, constrains logistic updates toward source coefficients, updates neural fusion heads with AnyPPG frozen, recalibrates ExtraTrees and fine-tunes the source-trained Net1D network. Transfer test patients do not enter fitting, adaptation, calibration or threshold selection.

AS and pre-TAVI capture use separate case totals. Procedure status and longitudinal observations do not enter AS classifier training. Morphology analyses use cross-sectional comparisons, a patient random-intercept pre/post model, paired follow-up tests and BH adjustment over the complete 510-feature family.

## Third-party model access

Obtain `load_anyppg/resnet1d.py` and its compatible checkpoint from [AnyPPG](https://github.com/Ngk03/AnyPPG) under upstream terms. Supply both paths to generate representations:

```bash
python -m pulseas extract --data <prepared-recordings> --cohort EDU --encoder-code <AnyPPG/load_anyppg/resnet1d.py> --checkpoint <anyppg-checkpoint.pth> --output outputs/features
```

The encoder remains frozen. The additional source-trained Net1D uses the compatible architecture initialised independently; pass `--encoder-code` when running transfer with Net1D. Architecture credit: [Shenda Hong's resnet1d](https://github.com/hsd1503/resnet1d). Reference hashes are in [upstream_hashes.json](upstream_hashes.json). Ext-PPG utilities for the optional raw MIMIC adapter are supplied separately. See [third-party notices](../THIRD_PARTY_NOTICES.md).

## Running and checking

`configs/study.yaml` contains study budgets and model settings. `configs/demo.yaml` reduces computation for synthetic data. The feature-level CPU demo and tests check runnable interfaces; optional GPU tests exercise neural and XGBoost paths.

Exact study results require authorised data, matching folds and transfer assignments, upstream weights and software environment. Synthetic results demonstrate the workflow and do not reproduce manuscript AUROCs.
