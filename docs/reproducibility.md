# Reproducibility and source mapping

This release extracts the AS analysis core from the study workspace into a portable package. It does not retrain the manuscript models during repository preparation. Source-file hashes and extracted function locations are in `source_manifest.json`; upstream implementation and checkpoint hashes are in `upstream_hashes.json`.

## Analysis provenance

| Public module | Study sources and retained behaviour |
|---|---|
| `preprocessing` | `ppg_quality_screening_v2.py`: EDU template SQI ≥0.90, amplitude stationarity, dropout and raw-device artefact checks. `select_mimic_ppg.py`: native 30-second Ext-PPG screening, three 10-second checks and seeded 15-second half selection. |
| `features` | `extract_operation_ppg_features_15s.py`, `run_pyppg_as_top_feature_classification.py`, `extract_mimic_extppg_features_20261005.py`: fixed fiducial adapter, within-segment z-score and exactly one pyPPG preprocessing pass. |
| `models`, `validation` | `reviewer_revision_preview_common.py`, `valve_classical.py`, `run_no_af_clinical_models_20260924.py`, `train_mimic_extppg_20261005.py`: model grids, cohort-specific fitting, Top-10 domain-diverse selection, patient-weighted MIMIC fits, aggregation and training-OOF Platt calibration. |
| `transfer` | `run_manuscript_transfer20_20261008.py` and its common model primitives: source-only development, 20/80 target split, tree continuation, anchored logistic update, prediction-head update and calibration-only ExtraTrees. The AS public runner has no dependency on AR experiment modules. |
| `evaluation` | `kent_clinical_models_20261013.py` and patient-level reporting utilities: five matched clinical baselines, complete cases, training-knot restricted cubic splines and paired AUROC differences. |
| `stratification` | `preview_clinical_ppg_enrichment_20261006.py`: calibrated OOF AS ranking, ceil counts, score ties, separate AS and pre-TAVI denominators. |
| `morphology` | `review_all_patient_mixed_morphology_20261009.py`, `build_reading_preview_morphology510_20261009.py` and complete-pulse template helpers: REML random intercept, Wald stage effect, paired follow-up, real-time templates and complete-family BH. |

Source wrappers for private identity mapping, registry reconciliation, cohort approvals, historical result promotion and manuscript formatting are deliberately not part of the public package. Their outputs are inputs to this analysis core. The repository makes no new assumption about missing valve grades or procedure status.

## Preserved experimental distinctions

1. **Internal development:** EDU and MIMIC are separate analyses. EDU input is patient-level; MIMIC PPG input is segment-level with equal total weight per patient. No patient or duplicate group crosses a fold.
2. **EDU folds:** the study's existing 5×3 assignments inherited joint-endpoint strata. Supply the frozen folds and patient order. Generating new AS-only folds is a new analysis, not an exact replay.
3. **MIMIC folds:** outer seed 17, inner seed `17 + outer_fold`; fitting seed `17 + outer_fold*100 + inner_fold`, final fold fit `17000 + outer_fold`.
4. **EDU fits:** retain the original SHA256-based seed recipe with base 42 and the original model slugs. Outer folds are numbered from one; MIMIC folds from zero. Platt fitting uses the recorded seed 42.
5. **External transfer:** seed 61007; source train/validation/internal-test 60/20/20; target adaptation is `ceil(0.20*N)` followed by a validation subset `ceil(0.20*N_adapt)` using seed+4. This replaces the historical 25% and 10% experiments. Those experiments are not part of this release.
6. **Adapter:** native MIMIC 125-Hz 15 seconds → complete signal resampled to 60 Hz → central 10 seconds → 125-Hz 1,250-sample z-scored encoder input. EDU starts from the corresponding 60-Hz input. Do not simplify the MIMIC adapter to a different direct crop/resample if exact replay is intended.
7. **Morphology:** 918 extracted feature slots and the 510-feature inference family are distinct. Each contrast's complete family includes unestimable tests as p=1 during BH adjustment. The representative figure panel does not define the correction family. Follow-up remains a paired Wilcoxon analysis and is not added to the pre/post mixed model.
8. **Calibration:** segment pre-calibration scores are aggregated before Platt calibration. Negative calibration slopes are retained; target-test labels are never used to reverse a score or choose a threshold. Sens@60%Spec refers to development-derived thresholds, not fixed test specificity.

## Checks and practical limits

The public tests exercise patient isolation, training-only feature selection, equal-patient aggregation, transfer role isolation, anchored-update gradients, source-tree preservation, neural-head source preservation, separate enrichment denominators and 510-feature correction. GPU tests are marked `gpu` and need the model extras and CUDA. The CPU demo is also checked in a separate environment.

Private local verification compares selected functions against the recorded source implementations and frozen inputs. Clinical input files, model outputs and patient-level verification material remain local. `release_checks.json`, when present, contains only non-identifying validation summaries.

The repository alone cannot reproduce exact manuscript cohort counts or AUROCs: those depend on the authorised frozen data, patient ordering, inherited roles, software and upstream weights. A successful synthetic demo establishes runnable interfaces, not clinical performance. Existing historical project-wide audit issues are not resolved by this packaging task.

The release uses CPU tests for portability and the recorded CUDA routes for study neural/XGBoost work. No automatic device fallback, encoder retraining, patient relabelling, post-test tuning or replacement of frozen manuscript results is performed by packaging.
