# Third-party notices

The MIT licence at the repository root applies to original PulseAS project code. It does not relicense third-party packages, architecture implementations, pretrained checkpoints or clinical datasets. No third-party source tree or model checkpoint is bundled here.

| Component | Upstream | Use |
|---|---|---|
| AnyPPG | https://github.com/Ngk03/AnyPPG | User-supplied frozen encoder implementation and pretrained checkpoint. Follow upstream access and licence terms. |
| Net1D / resnet1d | https://github.com/hsd1503/resnet1d | Architecture credited to Shenda Hong. The study uses the compatible implementation distributed with AnyPPG; it is loaded from a user-supplied file. |
| pyPPG | https://github.com/godamartonaron/GODA_pyPPG | Installed separately. PulseAS retains its project-specific complete-fiducial adapter and biomarker aggregation; upstream library terms remain applicable. |
| Ext-PPG | Native MIMIC-III-Ext-PPG signal processing utilities | Installed or obtained separately at the recorded upstream commit. The observed upstream licence is MIT, copyright AI4HealthUOL; retain the upstream licence with any separately obtained copy. |
| NeuroKit2 | https://github.com/neuropsychology/NeuroKit | EDU cleaning and template quality; MIT. |
| PyTorch | https://pytorch.org | Neural modules; BSD-style upstream terms. |
| XGBoost | https://github.com/dmlc/xgboost | Gradient-boosted trees; Apache-2.0. |
| NumPy, SciPy, pandas, scikit-learn, statsmodels, joblib, PyYAML | Respective upstream projects | Installed dependencies retain their respective licences. |

The original development workspace also contains rPPG-Toolbox and other third-party projects. Their complete directories, unrelated implementations, assets and licences are not copied into this compact release. Any downstream redistribution of those projects must follow their own terms.
