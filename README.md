# credit-transfer-benchmark

Replication package for:

> Zhang, Y., Fan, J., Liu, Y., Zhang, W., & Yan, C. Evaluating models of human choice when feedback format changes: Input-representation controls, identifiability checks and second-task tests. Manuscript submitted to *Behavior Research Methods*.

**Download:** the complete package (code, protocols, certificates, trained weights; 2,533 files) is attached to release [v1.0.0](https://github.com/niyaobuyaochibl/credit-transfer-benchmark/releases/tag/v1.0.0) as `replication_package.zip`. Unzip it and run `python replication_package/verify_package.py` to check every file against `MANIFEST.sha256`.

**Archived copy:** the same package is archived on Zenodo, [doi:10.5281/zenodo.23430698](https://doi.org/10.5281/zenodo.23430698) (stored in 5 MB parts; `join_parts.py` rebuilds and checks the zip).

Contact: Jingjing Fan (fjj1960@hebiace.edu.cn), School of Information Engineering, Hebei University of Architecture.

## Toolkit

[`toolkit/`](toolkit) contains `transfer_check`, a small NumPy-only Python package that implements the six-step checklist from the paper (Table 4) for any model that outputs choice probabilities, with tests and a worked example on simulated data that runs in a few seconds:

```
cd toolkit
python examples/two_format_bandit.py
```

See [`toolkit/README.md`](toolkit/README.md) for the steps and the functions that implement them.

## Licence

Code in this repository is released under the MIT licence (`LICENSE`). The public datasets keep the licences of their original authors and are not redistributed here.

## Replication package

Run `python verify_package.py` to check every file against `MANIFEST.sha256`. Supplementary Table S1 (`manuscript_20261015/supplement_S1.md`) maps every number in the paper to a certificate file and SHA-256 hash in this package.

### Study-specific reusable components

These files from the study can be adapted to other datasets without the study-specific models:

| File | What it does |
|---|---|
| `feedback_transfer_method_v1_20261012/study_statistics.py` | Estimands, participant-level paired bootstrap stratified by a grouping variable, calibration and the fixed baseline-selection rule, as pure array functions. |
| `feedback_transfer_method_v1_20261012/final_gate.py` | Fail-closed gate: checks every frozen asset against its hash, refuses to refreeze once held-out access has started, and binds the manifest to the access journal written at first access. |
| `feedback_transfer_method_v1_20261012/networks.py` | Causal predictors, including the summary-input control and the packet encoding of feedback. |
| `feedback_transfer_amendment_20261014/a1r5.py` | Recoverability study: refits candidate families on simulated learners with early stopping and reports a confusion matrix. |

### Layout

Directories keep the names used during the study, so that hashes recorded inside protocols and manifests still resolve.

| Directory | Contents |
|---|---|
| `feedback_transfer_method_v1_20261012/` | Main study: frozen protocol, models (`networks.py`), training, development selection, hash-gated held-out evaluation, synthetic studies, certificates (`final/results.json`, `final_manifest.json`) and trained weights (`fits/`). `task_b/` holds the independent-task study. |
| `feedback_transfer_amendment_20261014/` | Amendment A: recoverability (A-r5), behavioural revaluation test, closed-loop simulation, with each revision frozen before it ran. |
| `external_moran2019_20261014/` | Reproduction of Moran et al. (2019) and the multi-credit analogue test. |
| `capacity_sensitivity_20261016/` | Post hoc sensitivity analysis: summary-input GRU with as many parameters as the candidate; frozen before fitting, with a reproduction gate against the frozen held-out results. |
| `feedback_form_transfer_preparation_20261011/` | Cohort allocation, data interface, cognitive-model population fits. |
| `coarsened_feedback_*`, `feedback_event_learning_prototype_*`, `feedback_transfer_experiment_plan_*` | Earlier-stage code and protocols referenced by hash from the main protocol. |
| `within_person_feedback_public_triage_*`, `dispersed_feedback_*` | Scripts that download the public data from the original repositories, verify file hashes and build the event files used by the models. |
| `manuscript_20261015/` | Manuscript source, figure and supplementary-table scripts that re-verify certificate hashes before plotting. |

### Data

No raw participant data are redistributed. Obtain them from the original repositories (Maulhardt et al., 2026: https://osf.io/rp56b/ ; Cotet et al., 2026: https://osf.io/rnwxb/ ; Moran et al., 2019: the OSF deposit linked from the article) and rebuild the event files with the scripts above; the build scripts check source hashes.

`withheld_and_missing.json` lists, with frozen hashes, (i) derived files that contain participants' trial-by-trial choices and are withheld until the licence of the first dataset is confirmed, (ii) frozen artifacts that are not part of this package, and (iii) a few preflight and status records that were regenerated after the freeze, so the packaged copy differs from the frozen hash. These are verification records, not data, weights or results files.

### Environment

Python 3.13, PyTorch, NumPy, SciPy, threadpoolctl, Matplotlib. The main neural fits ran on one GPU instance; CPU execution is supported with one thread per process. Exact package versions are recorded in the `versions` field of each protocol.
