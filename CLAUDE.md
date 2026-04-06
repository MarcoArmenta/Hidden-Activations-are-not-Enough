# Hidden Activations Are Not Enough

## Overview

Research implementation of "Hidden Activations Are Not Enough: A General Approach to Neural Network Predictions" (arXiv:2409.13163) by Samuel Leblanc, Aiky Rasolomanana, and Marco Armenta. Computes "knowledge matrices" via quiver representations of neural networks and uses them as representations for adversarial example detection. Core claim: the same standard anomaly detectors, applied to knowledge matrices instead of penultimate-layer features, detect adversarial examples more consistently.

- **Language:** Python 3.11 + Bash (Slurm orchestration)
- **Cluster:** Compute Canada Alliance HPC (Rorqual, H100 GPUs)
- **License:** Apache 2.0
- **Key dependencies:** torch 2.2.2, torchvision 0.17.2, torchattacks 3.5.1, knowledgematrix (git+samueleblanc@0d26c7a), scikit-learn 1.3.2, scipy 1.10.1
- **Entry points:** `bash run_experiment.sh`, `bash calibration.sh`, `bash run_all_experiments.sh`

## TMLR Revision Context

Paper rejected by TMLR (Nov 2024). Three reviewers (Kv2C, BFn5, Vj72) + action editor (t5mo / Grigorios Chrysos). Core reframing for resubmission: **"representation comparison, not detector proposal"** — same standard detectors applied to knowledge matrices vs. penultimate features.

Reviewer concerns and how the codebase addresses each:

1. **Non-standard metrics** (Kv2C, Vj72) → AUROC / AUPR / FPR@95TPR for every cell (Step E)
2. **Unfair HP comparison** (BFn5) → 6 fixed-default detectors on all 3 representations (no per-detector tuning)
3. **Dimensionality confound** (BFn5, Vj72) → SVD rank ablation {16, 32, 64, 128, 256, 512}
4. **Weak baselines** (Kv2C, Vj72) → KNN, KDE, GMM, OCSVM, IsolationForest, Mahalanobis + Lee et al. (2018) multi-layer Mahalanobis
5. **Computational cost unreported** (Vj72) → wall-clock time + peak GPU memory instrumentation per representation
6. **Toy datasets** (BFn5) → CIFAR-10/100 with 4 CNN architectures
7. **Theorem 4.5 never validated** (BFn5) → Step G empirical validation of distance lower bound

## Directory Structure

```
├── run_experiment.sh          # Main pipeline orchestrator (generates + submits all Slurm jobs)
├── experiment_config.sh       # Shared config: accounts, resource profiles, helper functions
├── calibration.sh             # GPU calibration pipeline
├── run_all_experiments.sh     # Multi-experiment launcher
├── job_recovery.sh            # Manual recovery orchestrator
├── job_audit.sh               # Standalone data integrity audit
├── training.py                # Step A: model training
├── generate_matrices.py       # Step B: clean knowledge matrix computation (chunked)
├── generate_adversarial_examples.py  # Step C: adversarial example generation (per-attack)
├── generate_adversarial_matrices.py  # Step D: adversarial knowledge matrix computation (chunked)
├── compare_representations.py # Step E: 6 detectors × 3 representations + Lee baseline + SVD ablation
├── validate_theorem45.py      # Step G: empirical Theorem 4.5 validation
├── generate_latex_tables.py   # Step F: LaTeX table generation from results JSONs
├── calibrate.py               # GPU batch_size binary search + SLURM resource estimation
├── collect_errors.py          # Slurm log scanning + error aggregation → overall_errors.json
├── auto_resubmit.py           # OOM/timeout auto-retry with resource doubling
├── relaunch_sentinel.py       # Cyclic pipeline re-launch (up to MAX_SENTINEL_CYCLES)
├── pipeline_report.py         # Post-run analysis with error classification
├── isomorphism_experiment.py  # Isomorphism invariance demonstration
├── baselines/
│   └── lee2018.py             # Lee et al. (2018) multi-layer Mahalanobis baseline
├── constants/
│   └── constants.py           # All experiment configs, attack lists, architectures
├── matrix_construction/
│   └── parallel.py            # ParallelMatrixConstruction — chunked matrix computation with OOM retry
├── utils/
│   ├── utils.py               # get_architecture(), get_model(), get_dataset(), get_device()
│   ├── atomic_io.py           # atomic_torch_save(), atomic_json_dump() (write-to-tmp-then-rename)
│   ├── data_integrity.py      # verify_experiment(), zip verification
│   └── error_classification.py # Slurm error parsing regex patterns
├── docs/
│   ├── deep_code_review.md    # v1 code review (8 agents, 47 findings)
│   ├── deep_code_review_v2.md # v2 code review (15 agents, 8 critical + 21 high)
│   ├── deep_code_review_v3.md # v3 code review (post-bugfix validation)
│   ├── CHECKPOINTING.md       # Checkpointing and error handling details
│   ├── rebuttal/              # Revised TMLR manuscript (LaTeX source)
│   └── debate/                # Reviewer simulation and rebuttal planning
├── experiments/{experiment}/   # Per-experiment output (weights, matrices, results)
├── tables/                    # Generated LaTeX tables (Step F output)
├── slurm_out/, slurm_err/     # Slurm log files
└── gpu-monitor/               # Per-job GPU utilization logs
```

## Experimental Design

**Core claim: the same standard detectors, applied to knowledge matrices instead of penultimate-layer features, detect adversarial examples more consistently.**

| Dimension | Values | Count |
|-----------|--------|-------|
| Representations | Penultimate activations, Knowledge matrices, SVD-reduced matrices | 3 |
| Detectors | KNN, KDE, GMM, OCSVM, IsolationForest, Mahalanobis | 6 |
| Attacks | See ATTACKS in `constants/constants.py` | 16 |
| Experiments | alexnet_cifar10, resnet_cifar10, resnet_cifar100, vgg_cifar100 | 4 |

**Metrics:** AUROC (primary), AUPR, FPR@95TPR — all three reported for every cell in the 6×3 grid.

**SVD ablation:** ranks {16, 32, 64, 128, 256, 512}. Default rank 256 when raw dimensionality exceeds 256. Flag: `--svd_ablation`.

**Lee et al. (2018):** Multi-layer Mahalanobis baseline (`baselines/lee2018.py`). Runs separately from the 6×3 grid — uses its own logistic regression combiner across layers.

**16 attacks** across 5 categories: gradient-based (7), AutoAttack ensemble (4), gradient-free (3), elastic-net (2), baseline noise (1). Note: Square appears in both AutoAttack and gradient-free categories.

## System Integration & Data Flow

### Configuration Flow

Configuration originates in three layers: (1) shell variables in `experiment_config.sh`, (2) Python constants in `constants/constants.py`, (3) optional `calibration.json` overrides. The orchestrator bakes all resolved values into heredoc Slurm scripts — no runtime config lookup. Python workers receive config via CLI arguments + `constants.py` imports.

Resource resolution cascade (each layer overrides previous):
1. Shell defaults (`experiment_config.sh:33-67`)
2. Calibration override (`run_experiment.sh:1717-1727` reads `calibration.json`)
3. Checkpoint retry (`run_experiment.sh:403-423`, exit_code 137 → `double_mem()`)
4. sacct detection (`experiment_config.sh:189-226`, OOM_KILLED → `double_mem()`)
5. Error report (`run_experiment.sh:337-348`, `overall_errors.json` OOM entries → `double_mem()`)

### Data Flow Between Pipeline Stages

| From | To | Artifact | Format | Path |
|------|----|----------|--------|------|
| Step A | B, C, D, E, G | Trained weights | PyTorch .pth | `experiments/{exp}/weights/epoch_{N}.pth` |
| Step B ×8 | E | Clean KMs | Zip of matrix.pt files | `experiments/{exp}/matrices_task_{0-7}.zip` |
| Step C ×17 | D, E | Adv examples | PyTorch .pth tensors | `experiments/{exp}/adversarial_examples/{attack}/*.pth` |
| Step D ×8 | E | Adv KMs | Zip of matrix.pth files | `experiments/{exp}/adv_matrices_task_{0-7}.zip` |
| Step E | F | Comparison results | JSON | `experiments/{exp}/comparison/representation_comparison.json` |
| Step G | F | Theorem validation | JSON | `experiments/{exp}/theorem45/theorem45_results.json` |
| Error Scan | Sentinel, next run | Error report | JSON v3.0 | `experiments/{exp}/overall_errors.json` |

### Error Propagation

Pipeline uses `afterok` Slurm dependencies — any non-zero exit from an upstream job cancels all downstream jobs. Error Scan and Sentinel run with `afterany` to ensure self-recovery. Three-level retry: (1) Python in-process batch_size halving, (2) orchestrator checkpoint-based resource doubling, (3) `auto_resubmit.py` reading `overall_errors.json`. Sentinel can re-invoke the entire pipeline up to 5 cycles.

### Architecture Diagram

```
                        ┌─────────────────────────┐
                        │   Step A: training.py    │
                        │  Produces: weights/*.pth │
                        └─────┬──────┬──────┬─────┘
                              │      │      │
                     ┌────────┘      │      └────────┐
                     ▼               ▼               ▼
   ┌──────────────────────┐ ┌────────────────┐ ┌─────────────────────┐
   │ Step B ×8 chunks     │ │ Step C ×17     │ │ Step G: validate    │
   │ generate_matrices.py │ │ (per attack)   │ │ _theorem45.py       │
   │ → matrices_task_N.zip│ │ gen_adv_ex.py  │ │ → theorem45_results │
   └──────────┬───────────┘ │ → adv_examples │ └──────────┬──────────┘
              │             └───────┬────────┘            │
              │                     │                     │
              │          ┌──────────┘                     │
              │          ▼                                │
              │ ┌──────────────────────┐                  │
              │ │ Step D ×8 chunks     │                  │
              │ │ gen_adv_matrices.py  │                  │
              │ │ → adv_matrices_N.zip │                  │
              │ └──────────┬───────────┘                  │
              ▼            ▼                              │
   ┌──────────────────────────────────┐                   │
   │ Step E: compare_representations  │                   │
   │ → representation_comparison.json │                   │
   └──────────────┬───────────────────┘                   │
                  │                                       │
                  └──────────────┬────────────────────────┘
                                 ▼
   ┌──────────────────────────────────┐
   │ Step F: generate_latex_tables.py │
   │ → tables/*.tex                   │
   └──────────────┬───────────────────┘
                  ▼
   Audit (afterok) → Error Scan (afterany) → Sentinel (afterany) ─╮
                                                                    │
   ╭──────────────────────────────────────────────────────────────╯
   │  If retryable errors remain & cycle < MAX_SENTINEL_CYCLES:
   ╰──→ re-invokes run_experiment.sh (skips completed, doubles resources)
```

## Pipeline

### Main Pipeline (run_experiment.sh)

7-step Slurm pipeline. Step IDs encode dependency depth.

**Quick start:**
```bash
bash calibration.sh                              # Run first
bash run_experiment.sh --skip-audit alexnet_cifar10  # Then run pipeline
```

**Prerequisites:** (1) Calibration completed, (2) datasets pre-downloaded in `data/`, (3) pretrained weights in `experiments/{arch}_imagenet/weights/`, (4) Python venv at `env/`

**Job execution order:**

1. **Step A** (Training) — trains CNN model — depends on: nothing
2. **Step B ×8** (Clean Matrices) — computes knowledge matrices for training data — depends on: A
3. **Step C ×17** (Adversarial Examples) — generates adversarial examples per attack — depends on: A
4. **Step D ×8** (Adversarial Matrices) — computes knowledge matrices for adversarial examples — depends on: A + all C
5. **Step E** (Representation Comparison) — 6 detectors × 3 representations + SVD ablation + Lee baseline — depends on: A + all B + all C + all D
6. **Step G** (Theorem 4.5 Validation) — empirical distance lower bound validation — depends on: A only
7. **Step F** (LaTeX Tables) — generates paper tables from JSON results — depends on: E + G
8. **Final Audit** — verifies all pipeline artifacts — depends on: E + G + F (afterok)
9. **Error Scan** — aggregates errors, runs auto_resubmit — depends on: ALL (afterany)
10. **Relaunch Sentinel** — re-invokes pipeline if retryable errors remain — depends on: Error Scan (afterany)

### Configuration

| Variable | Default | Description | File |
|----------|---------|-------------|------|
| `ACCOUNT` | `def-assem` | SLURM billing account | `experiment_config.sh:14` |
| `TOTAL_CHUNKS` | `8` | Parallel chunks for matrix computation | `experiment_config.sh:17` |
| `BATCH_SIZE` | `1800` | KM columns per GPU pass | `experiment_config.sh:18` |
| `NUM_SAMPLES_PER_CLASS` | `500` | Training samples per class (Step B) | `experiment_config.sh:19` |
| `SAMPLES_PER_ATTACK` | `500` | Adv examples per attack (Step D) | `experiment_config.sh:20` |
| `MAX_SENTINEL_CYCLES` | `5` | Max pipeline re-launch cycles | `experiment_config.sh:31` |
| `SAVE_INTERVAL` | `200` | Incremental save every N matrices | `experiment_config.sh:27` |
| `SAVE_GRACE_SECONDS` | `180` | Emergency save seconds before wall-time | `experiment_config.sh:29` |

**Per-step SLURM resources (normal / test mode):**

| Step | GPU | CPUs | Time | Memory |
|------|-----|------|------|--------|
| A (Training) | H100 ×1 | 2 | 6h / 10m | 15G / 8G |
| B (Matrices ×8) | H100 ×1 | 12 / 4 | 20m / 15m | 280G / 32G |
| C (Adv Examples) | H100 ×1 | 4 | 3h / 30m | 32G |
| D (Adv Matrices ×8) | H100 ×1 | 12 / 4 | 12h / 30m | 280G / 32G |
| E (Comparison) | H100 ×1 | 8 / 4 | 8h / 1h | 128G / 16G |
| G (Theorem 4.5) | H100 ×1 | 4 | 6h / 30m | 64G / 16G |
| F (LaTeX) | none | 2 | 15m / 10m | 4G / 2G |

### Launch Modes

```bash
# Full pipeline with pre-flight audit
bash run_experiment.sh alexnet_cifar10

# Skip audit, direct submission
bash run_experiment.sh --skip-audit alexnet_cifar10

# Quick test (small samples, short times)
bash run_experiment.sh --test --skip-audit alexnet_cifar10

# Dry run (generate scripts only, no submission)
bash run_experiment.sh --dry-run --test alexnet_cifar10

# All experiments
bash run_all_experiments.sh
```

### Manual Step Execution

```bash
python training.py --experiment_name alexnet_cifar10 --temp_dir $SLURM_TMPDIR
python generate_matrices.py --experiment alexnet_cifar10 --chunk_id 0 --total_chunks 8 --batch_size 1800
python generate_adversarial_examples.py --experiment_name alexnet_cifar10 --temp_dir $SLURM_TMPDIR
python generate_adversarial_matrices.py --experiment_name alexnet_cifar10 --chunk_id 0 --total_chunks 8
python compare_representations.py --experiment alexnet_cifar10 --temp_dir $SLURM_TMPDIR --svd_ablation
python validate_theorem45.py --experiment alexnet_cifar10 --temp_dir $SLURM_TMPDIR
python generate_latex_tables.py --output tables/
```

### Monitoring & Recovery

```bash
python pipeline_report.py --experiment alexnet_cifar10       # Post-run report
python collect_errors.py --experiment alexnet_cifar10         # Error collection
bash job_recovery.sh alexnet_cifar10                          # Manual recovery
python -m pytest unit_test/                                   # Unit tests
```

## Checkpointing & Error Handling

### Checkpoints

Three-tier system: (1) per-step JSON files at `experiments/{exp}/checkpoints/step_{X}.json` written by Slurm jobs on completion/failure, (2) Python-level file-existence checks skip already-computed individual matrices, (3) training PyTorch checkpoints every 10 epochs.

- **Format:** JSON (`{"status": "complete"|"partial"|"failed", "exit_code": int, "mem": "XG", "time": "HH:MM:SS", "timestamp": str}`)
- **Save trigger:** End of each Slurm job (success or failure)
- **Path:** `experiments/{experiment}/checkpoints/step_{A|B_chunk_N|C_attack_NAME|D_chunk_N|E|F|G}.json`
- **Resume:** Orchestrator reads checkpoint via `read_checkpoint_status()`. Complete + output exists → SKIP. Partial → resubmit (Python resumes via file-existence). Failed → double resources + resubmit.

Steps B and D also have **incremental save** (background process every 60s) and **emergency save** (`--signal=B:USR1@180` trap before wall-time kill). All file writes use atomic tmp+rename (`utils/atomic_io.py`).

### OOM Retry (3 levels)

1. **Python in-process** (`parallel.py:192`, `generate_adversarial_matrices.py:84`): halves `batch_size`, up to 4 attempts
2. **Orchestrator checkpoint** (`run_experiment.sh`): reads exit_code 137, calls `double_mem()` (cap 480G)
3. **auto_resubmit.py**: reads `overall_errors.json`, doubles `--mem` in Slurm script, max 2 retries per step

### Timeout Retry (2 levels)

1. **Orchestrator checkpoint**: reads exit_code 140, calls `double_time()` (cap 48h)
2. **auto_resubmit.py**: same mechanism as OOM

### Error Logging

- **Slurm logs:** `slurm_out/PIPE_{STEP}_{experiment}[_c{chunk}|_{attack}]_{jobid}.out/.err`
- **GPU monitoring:** `gpu-monitor/{experiment}.{STEP}.{chunk}.log` (nvidia-smi every 30s)
- **Global aggregation:** `collect_errors.py` → `experiments/{exp}/overall_errors.json` (schema v3.0)
- **Error types detected:** CUDA OOM, system OOM, timeout, CUDA error, network error, missing file, module error, zip corruption, assertion, permission
- **Python logging:** Not used — all output via `print()` to Slurm stdout

## Experiment Configuration

Defined in `constants/constants.py`:

- `DEFAULT_EXPERIMENTS`: dict mapping experiment names to hyperparams (epochs, lr, batch_size, architecture_index, dataset, optimizer, scheduler)
- `ATTACKS`: list of **16** adversarial attack names: GN, FGSM, PGD, EOTPGD, MIFGSM, VMIFGSM, CW, DeepFool, APGD, APGDT, FAB, Square, Pixle, SPSA, EADL1, EADEN
- `ATTACK_CATEGORIES`: gradient_based (7), autoattack (4), gradient_free (3), elastic_net (2), baseline_noise (1)
- `ARCHITECTURES`: indexed by negative integers — -4=LeNet, -3=AlexNet, -2=ResNet, -1=VGG

Main experiments: `alexnet_cifar10`, `resnet_cifar10`, `resnet_cifar100`, `vgg_cifar100`

## Output Structure

```
experiments/{experiment}/
├── weights/epoch_{N}.pth                       # Step A
├── weights/history.json                        # Training curves
├── matrices_task_{0-7}.zip                     # Step B (chunked)
├── adversarial_examples/{attack}/              # Step C
│   ├── adversarial_examples.pth
│   └── wrong_predictions.pth
├── adv_matrices_task_{0-7}.zip                 # Step D (chunked)
├── comparison/representation_comparison.json   # Step E
├── comparison/svd_ablation.json                # Step E --svd_ablation
├── theorem45/theorem45_results.json            # Step G
├── calibration.json                            # GPU calibration
├── overall_errors.json                         # Error aggregation
├── audit_report.json                           # Data integrity audit
├── auto_resubmit_status.json                   # Retry status
├── checkpoints/                                # Per-step completion tracking
│   ├── step_A.json
│   ├── step_B_chunk_{0-7}.json
│   ├── step_C_attack_{NAME}.json
│   ├── step_D_chunk_{0-7}.json
│   ├── step_E.json, step_F.json, step_G.json
│   └── cycle_count.txt                         # Sentinel cycle counter
└── orchestrator_jobs/                          # Generated Slurm scripts
tables/*.tex                                    # Step F output
slurm_out/, slurm_err/                          # Slurm logs
gpu-monitor/                                    # GPU utilization logs
reports/                                        # Pipeline reports
```

## Critical Patterns

### KnowledgeMatrixComputer expects 3D input
The `knowledgematrix` library's `KnowledgeMatrixComputer.forward()` expects **3D tensors** `(C, H, W)`, NOT 4D `(1, C, H, W)`. Never use `.unsqueeze(0)` before calling `forward()`.

### No internet on compute nodes
Compute nodes cannot download anything. All model creation must use `pretrained=False`. Pretrained weights loaded from local files. Datasets must be pre-downloaded on login nodes.

### torchattacks version compatibility
Cluster has torchattacks 3.3.0, local uses 3.5.1. Use lazy `getattr(torchattacks, cls_name, None)` to handle missing attack classes.

### freeze_features compatibility
The `knowledgematrix` package's AlexNet may not accept `freeze_features` kwarg. `utils/utils.py:get_architecture()` uses `inspect.signature` to check before passing it.

### Matrix file extensions
Training matrices use `.pt` extension, adversarial matrices use `.pth`. This is consistent within each producer but creates two conventions.

### $SLURM_TMPDIR data staging
All GPU jobs copy data from persistent storage (`$SLURM_SUBMIT_DIR`) to fast local SSD (`$SLURM_TMPDIR`) at job start, and copy results back at job end.

## Environment

- **Cluster modules:** `StdEnv/2023 python/3.11.5 scipy-stack/2025a`
- **Virtual env:** `env/` (created via `python -m venv env`)
- **Cluster deps:** `requirements-slurm.txt` (torch 2.2.2)
- **Local deps:** `requirements-local.txt` (torch 2.6.0)
- **Key package:** knowledgematrix (`git+https://github.com/samueleblanc/knowledgematrix.git@0d26c7a`)

## Known Issues & Gotchas

### Critical (Scientific Validity)

| ID | Location | Description |
|----|----------|-------------|
| NC1 | `baselines/lee2018.py:162-189` | Lee et al. input preprocessing is a no-op — gradient graph severed by numpy conversion; epsilon perturbation never fires |
| NC2 | `compare_representations.py:599,607` | Clean test data differs between representations — penultimate/all-layer use random 2000-sample subset; KMs use different subset |
| NC3 | `compare_representations.py:645-654` | Adversarial sample count mismatch — 2000 for penultimate/all-layer vs ~200 for KMs |
| NC4 | `training.py:237-259` | Epoch-60 unfreeze recreates optimizer with lr=1e-5 — nukes LR for from-scratch training (resnet_cifar10, resnet_cifar100) |
| NC5 | `validate_theorem45.py:329-330` | Theorem 4.5 bound satisfaction is tautological — gamma=min(d_M/d_f) guarantees 100% by construction |
| CR-1 | `generate_latex_tables.py:900` | KeyError on `bound_satisfaction_rate` — field renamed in v2 bugfix but table generator reads old key |
| CR-2 | `run_experiment.sh` vs `auto_resubmit.py` | Step naming mismatch — shell uses letter IDs (A-G), auto_resubmit uses numeric IDs (1, 2a, 2b...) — entire auto-retry system inoperable |
| CR-3 | `compare_representations.py` | Clean test samples still differ between representations after v2 fix — random.sample with same seed but different k |

### High Priority

| ID | Location | Description |
|----|----------|-------------|
| NH1 | `compare_representations.py:738` | Lee eval_clean_scores uses wrong LR model across attack iterations |
| NH3 | `compare_representations.py:274` | KDE bandwidth=1.0 hardcoded — undermines "fair comparison" claim |
| NH4 | `compare_representations.py:317` | GMM full covariance severely underdetermined (330K params from 5K samples) |
| NH5 | `compare_representations.py:170-237` | Per-class Mahalanobis degenerate with 50 samples/class (CIFAR-100) |
| NH13 | All Python entry points | No global random seed — results not reproducible |
| C5 | `constants/constants.py:43` | ATTACKS has 16 entries; spec claims 17. Square double-counted across categories |

### Infrastructure

- Training `--from_checkpoint` not passed by orchestrator — partial training loses all progress on restart
- `double_mem()` leaks across chunks — one chunk OOM doubles memory for ALL subsequent chunks
- Steps A, C, E, G, F have NO signal handling — OOM/timeout kills leave no checkpoint
- `job_recovery.sh` has hardcoded wrong GPU type and wrong venv name (NH12)
- `.gitignore` is malformed — bare tokens like "json", "os", "torch" match unintended files

**Suggested fix priority:** NC1 → NC4 → NC2+NC3 → CR-2 → NC5 → NH13 → C5 → NH3 → NH4

See `docs/deep_code_review_v2.md` and `docs/deep_code_review_v3.md` for complete reports.
