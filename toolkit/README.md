# transfer_check

A small NumPy-only package for checking whether a model of human choice still predicts behaviour after the feedback format changes, for example from component-wise to summed outcome reports. Each module implements one or more steps of the checklist in Table 4 of the accompanying paper (Zhang, Fan, Liu, Zhang & Yan, submitted to *Behavior Research Methods*).

The package does not fit models. You bring your own models and their predicted choice probabilities; the package handles the parts of the evaluation that are easy to get wrong.

## Install and test

```
pip install .            # or just put this folder on PYTHONPATH; the only dependency is NumPy
pip install pytest && pytest tests
python examples/two_format_bandit.py
```

## The six steps

| Step | What to do | Function |
|---|---|---|
| 1. Define the shift | Write each format as an observation matrix `F` (observed x latent outcome contributions) and report each transfer direction separately. `common_observation` returns the outcome information that every format reveals. | `common_observation`, `project`; directions in `transfer_contrast` |
| 2. Add a shift-invariant control | Include a model that reads only the features from `common_observation`, so that its input is identical under every format. Match the candidate's parameter count as well. | `project` |
| 3. Fix the comparison before testing | Pick the baseline on development people by a fixed rule, plan the number of held-out people for the smallest effect of interest, freeze code, fitted parameters and the selection, then open the held-out data once. The primary contrast is a person-level paired bootstrap stratified by design (for example, which format came first). | `select_baseline`, `Gate`, `transfer_contrast` |
| 4. Check identifiability | Simulate learners with known mechanisms under each format, fit every candidate and see whether each mechanism is recovered. Models with identical predictions are reported as ties. | `recoverability` |
| 5. Make effects interpretable | Report per-person counts and calibration next to mean negative log-likelihood, and convert per-choice differences into nats per session. | `person_scores`, `calibration`, `per_session` |
| 6. Test the winner on a second task | Refit all models on a task with a different observation structure and check where the transfer winner ranks before treating it as the reference. | (your models; scored with `person_scores`) |

Conventions: choice arrays are people x trials with 0, 1, or -1 for a missing choice; probabilities are for choice 1; contrasts are baseline loss minus candidate loss, so positive values favour the candidate.

## Worked example

`examples/two_format_bandit.py` runs all six steps on simulated data in a few seconds. Each choice yields an immediate outcome and a delayed outcome that arrives on the next trial; component-wise feedback shows both, summed feedback shows only the total. Simulated people are half multi-credit learners (they credit delayed outcomes to the choice that caused them) and half single-credit learners. Three Q-learning models are compared: the candidate `multi_credit`, `single_credit`, and the shift-invariant control `summed_only`.

The output (seed fixed in the script) illustrates why each step matters:

- **Steps 1 and 3.** The shared feature is the total, `[[1, 1]]`. `single_credit` is selected on development people and beats the candidate on held-out people by 0.084 nats per choice (95% CI 0.029 to 0.150), but the difference is three times larger in one direction (component to summed: 0.130) than in the other (0.038, interval includes zero).
- **Step 4.** Under component-wise feedback all three mechanisms are recovered. Under summed feedback `single_credit` and `summed_only` make identical predictions and cannot be told apart, so a comparison between them on summed data says nothing about the mechanism.
- **Step 5.** The mean effect corresponds to about 12.5 nats per 150-trial session, yet 15 of 30 held-out people favour the candidate: the population is a mixture, and the mean hides it.
- **Step 6.** On a second task (two-trial delay, component-wise feedback), the transfer winner `single_credit` ranks last of three.

The script writes a full report, the frozen manifest and the access journal to `examples/output/`.

## Reproducing the paper's primary contrast

`examples/reproduce_paper.py` reads `examples/paper_heldout_contrasts.json`, which holds, for each of the 28 held-out participants, the session order and the two directional contrasts NLL(B\*) − NLL(multi-credit) from the paper's frozen results certificate (its SHA-256 is recorded in the file; no trial-level data are included). With the paper's bootstrap seed, `transfer_contrast` reproduces the primary contrast (−0.0252, 95% CI −0.0328 to −0.0178), both directional contrasts and the failed decision rule exactly:

```
python examples/reproduce_paper.py
```

## Limits

The `Gate` documents the order of events (what was frozen, and when held-out data were first opened); it cannot prove that nobody looked at the held-out data beforehand. The bootstrap intervals are percentile intervals with people as the resampling unit. The example models are deliberately simple and are not the models of the paper; the paper's own code is in the replication package.

## Licence

MIT; see `LICENSE`.
