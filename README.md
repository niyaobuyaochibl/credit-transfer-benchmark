# credit-transfer-benchmark

Replication package for:

> Zhang, Y., Fan, J., Liu, Y., Zhang, W., & Yan, C. Testing whether models of human learning generalize across feedback formats: A checklist, a worked example and software. Manuscript submitted to *Behavior Research Methods*.

An earlier version of the manuscript had the title "Evaluating models of human choice when feedback format changes: Input-representation controls, identifiability checks and second-task tests"; the data, the fitted models and the confirmatory results are the same in both versions.

**Download:** the complete package (code, analysis plans, result files, fitted models; 2,575 files) is attached to release [v1.0.4](https://github.com/niyaobuyaochibl/credit-transfer-benchmark/releases/tag/v1.0.4) as `replication_package.zip`. Unzip it and run `python replication_package/verify_package.py` to check every file against `MANIFEST.sha256`.

**Archived copy:** the same package is archived on Zenodo, [doi:10.5281/zenodo.23430697](https://doi.org/10.5281/zenodo.23430697) (this DOI always resolves to the latest version; stored in 5 MB parts, `join_parts.py` rebuilds and checks the zip).

Contact: Jingjing Fan (fjj1960@hebiace.edu.cn), School of Information Engineering, Hebei University of Architecture.

## Toolkit

[`toolkit/`](toolkit) contains `transfer_check`, a small NumPy-only Python package that implements the general computations of the six-step checklist from the paper (Table 1) for any model that outputs choice probabilities, with tests and a worked example on simulated data that runs in a few seconds:

```
cd toolkit
python examples/two_format_bandit.py
```

See [`toolkit/README.md`](toolkit/README.md) for the steps and the functions that implement them. The package does not fit models, and the behavioral check of Step 5 depends on the task; the script used in the paper is `manuscript_brm_r1_20261112/behavior_check.py` in the replication package.

## Licence

Code in this repository is released under the MIT licence (`LICENSE`). The public data sets keep the licences of their original authors and are not redistributed here.

## Replication package

The sections below describe the contents of `replication_package.zip`.

Run `python verify_package.py` to check every file against `MANIFEST.sha256`. Supplementary Table S1 (`manuscript_brm_r1_20261112/supplement_S1.md`) links every number in the paper to a result file and its SHA-256 checksum in this package.

### Study-specific reusable components

These files can be adapted to other data sets without the models of this study (see the checklist in Table 1 of the paper):

| File | What it does |
|---|---|
| `feedback_transfer_method_v1_20261012/study_statistics.py` | Estimands, paired bootstrap over participants stratified by a grouping variable, calibration and the fixed rule for selecting the benchmark, as pure array functions. |
| `feedback_transfer_method_v1_20261012/final_gate.py` | Lock on the test data: checks every fixed file against its checksum, refuses changes once the test data have been opened, and ties the file list to the access record written at first access. |
| `feedback_transfer_method_v1_20261012/networks.py` | The network models, including the network that reads only the total feedback and the labeled encoding of feedback. |
| `feedback_transfer_amendment_20261014/a1r5.py` | Model recovery: refits the candidate models on simulated learners with early stopping and tabulates which candidate predicts best. |
| `manuscript_brm_r1_20261112/behavior_check.py` | Exploratory analysis of repeated choices (Step 5 of the checklist): the same logistic regression for participants' choices and for the probabilities predicted by each model, with a bootstrap over participants. |

### Layout

Directories keep the names used during the study, so that checksums recorded inside the analysis plan and the file lists still resolve.

| Directory | Contents |
|---|---|
| `feedback_transfer_method_v1_20261012/` | Main study: analysis plan, models (`networks.py`), fitting, selection on the development participants, the locked evaluation on the test participants, simulations, result files (`final/results.json`, `final_manifest.json`) and fitted models (`fits/`). `task_b/` holds the study on the second task. |
| `feedback_transfer_amendment_20261014/` | Amendment A: model recovery (A-r5), a planned behavioral test of retrospective revaluation that failed its validity check and was not applied to human data, and a simulation of recommendation policies that the paper does not report; each revision was fixed before it ran. |
| `external_moran2019_20261014/` | Reproduction of Moran et al. (2019) and the test of the multi-credit mechanism on those data. |
| `capacity_sensitivity_20261016/` | Exploratory: the selected benchmark refitted with as many parameters as the multi-credit model; fixed before fitting. |
| `feedback_form_transfer_preparation_20261011/` | Allocation of participants to the fitting, stopping, development and test groups; data interface; fits of the cognitive models. |
| `coarsened_feedback_*`, `feedback_event_learning_prototype_*`, `feedback_transfer_experiment_plan_*` | Earlier code and plans that the main analysis plan refers to by checksum. They include the definitions of the cognitive models (`coarsened_feedback_classical_probe_20261004/classical_probe.py`, `coarsened_feedback_strong_controls_20261004/models.py`). |
| `within_person_feedback_public_triage_*`, `dispersed_feedback_*` | Scripts that download the public data from the original repositories, verify file checksums and build the event files used by the models. |
| `manuscript_brm_r1_20261112/` | Current manuscript (`manuscript.md`), supplementary tables S1 to S8 and their script (`build_supplement.py`), the rest of the supplement (`supplement_notes.md`: Supplementary Tables S9 to S11, Supplementary Notes S1 to S7 and Supplementary Figures S1 to S3), figures and their scripts (`make_figures.py` and `behavior_check.py`; in `figures/`, `fig1` to `fig3` are Figures 1 to 3 of the paper, `fig5_simulated` is Figure 4, and `fig4_calibration`, `fig5_stress` and `fig6_second_task` are Supplementary Figures S1 to S3), the results of the analysis of repeated choices (`behavior_check.json`, aggregates only; console output in `behavior_check_run.log`), the exploratory within-person test of the cognitive models (`within_person_check.py`; results in `within_person_check.json`, which holds NLL per participant but no trial data; console output in `within_person_run.log`; `within_person_plan_sha.txt` is the checksum of the script recorded before its first run, and the script, output and log of that first run are kept as `within_person_check_first_run.py`, `within_person_check_first_run.json` and `within_person_first_run.log`) and a consistency check of the numbers in the text (`check_manuscript.py`). `baseline_submitted_20261019/` holds the text of the first submission that this check compares against. |
| `manuscript_20261015/` | First submission: manuscript source, figure and table scripts. Kept because the scripts of the current version are derived from it. |

### Terms

Code and result files keep the names used during the study. The paper uses the following terms for them.

| In code and result files | In the paper |
|---|---|
| `disjoint` / `conjoint` (component-wise / summed) | separate / summed feedback |
| `conjoint` source, `disjoint` target (S to C) | summed-to-separate direction |
| held-out people | test participants |
| `B*`, `total_gru` | selected benchmark (total-coded network) |
| `parent_common_sum` | total-only network |
| `parent_full` | full-feedback network |
| `packet_gru` | labeled-feedback network |
| `gaussian_neural` | Bayesian credit model |
| `point_credit`, `multi_credit` | single-credit, multi-credit model |
| `parent_strong_union` | combined cognitive model |

### Data

No raw participant data are redistributed. Obtain them from the original repositories (Maulhardt et al., 2026: https://osf.io/rp56b/ ; Cotet et al., 2026: https://osf.io/rnwxb/ ; Moran et al., 2019: the OSF deposit linked from the article) and rebuild the event files with the scripts above; the build scripts check the checksums of the source files.

`withheld_and_missing.json` lists, with checksums, (i) derived files that contain participants' trial-by-trial choices and are withheld until the licence of the first data set is confirmed, (ii) fixed files that are not part of this package, (iii) a few status records that were regenerated after the analysis plan was fixed, so that the packaged copy differs from the recorded checksum, and (iv) the 40 files with the models' predictions for the test participants, which also contain those participants' choices and are withheld for the same reason. `manuscript_brm_r1_20261112/behavior_check.py` and `manuscript_brm_r1_20261112/within_person_check.py` need the files under (iv) and the event file of the first data set; they verify each of them against its recorded checksum before use. `build_supplement.py` and `check_manuscript.py` run on the files in this package.

### Environment

Python 3.13, PyTorch, NumPy, SciPy, threadpoolctl, Matplotlib. The main network fits ran on one GPU instance; CPU execution is supported with one thread per process. Exact package versions are recorded in the `versions` field of each analysis plan, and those of the analysis of repeated choices and of the within-person test in `behavior_check.json` and `within_person_check.json`.
