#!/bin/bash
#SBATCH --account=def-assem
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm_out/AUDIT_%A.out
#SBATCH --error=slurm_err/AUDIT_%A.err

# ==============================================================
# job_audit.sh — Full experiment audit
#
# Audits all artifacts for an experiment, prints a color-coded
# report, saves a JSON report, and generates a recovery plan.
# ==============================================================

# --- User-configurable variables ---
export EXPERIMENT="alexnet_cifar10"
export TOTAL_CHUNKS=8
ENV_NAME="env"

set -euo pipefail
echo "=== Audit starting on $(hostname) at $(date) ==="

# --- Environment ---
mkdir -p $SLURM_SUBMIT_DIR/slurm_out
mkdir -p $SLURM_SUBMIT_DIR/slurm_err

echo "Loading modules..."
module load StdEnv/2023 python/3.11.5 scipy-stack/2025a || true

echo "Activating venv ($ENV_NAME)..."
if [ -d "$SLURM_SUBMIT_DIR/$ENV_NAME" ]; then
    source $SLURM_SUBMIT_DIR/$ENV_NAME/bin/activate
else
    echo "ERROR: venv '$ENV_NAME' not found at $SLURM_SUBMIT_DIR/$ENV_NAME" >&2
    exit 1
fi

# Prevent torch from probing GPUs on CPU-only nodes
export CUDA_VISIBLE_DEVICES=""

# --- Run audit and save JSON report ---
cd $SLURM_SUBMIT_DIR
echo "Running audit..."

python << 'AUDIT_EOF'
import sys
import os
import json

sys.path.insert(0, os.environ["SLURM_SUBMIT_DIR"])

from utils.data_integrity import verify_experiment, _print_report
from constants.constants import ATTACKS, DEFAULT_EXPERIMENTS

experiment = os.environ["EXPERIMENT"]
total_chunks = int(os.environ["TOTAL_CHUNKS"])

experiment_dir = os.path.join(os.environ["SLURM_SUBMIT_DIR"], "experiments", experiment)

# Determine num_classes from the experiment's dataset
num_classes = 10
num_samples_per_class = 1000
num_samples_rejection_level = 10000

if experiment in DEFAULT_EXPERIMENTS:
    dataset = DEFAULT_EXPERIMENTS[experiment].get("dataset", "cifar10")
    if dataset == "cifar100":
        num_classes = 100
    elif dataset == "imagenet":
        num_classes = 1000

report = verify_experiment(
    experiment_dir=experiment_dir,
    experiment_name=experiment,
    num_classes=num_classes,
    num_samples_per_class=num_samples_per_class,
    total_chunks=total_chunks,
    num_samples_rejection_level=num_samples_rejection_level,
    attacks_list=ATTACKS,
    sample_ratio=0.1,
)

_print_report(report)

report_path = os.path.join(experiment_dir, "audit_report.json")
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
print(f"\nJSON report saved to: {report_path}")
AUDIT_EOF

echo "Audit complete. Generating recovery plan..."

# --- Generate recovery plan ---
python << 'RECOVERY_PLAN_EOF'
import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.environ["SLURM_SUBMIT_DIR"])

experiment = os.environ["EXPERIMENT"]
total_chunks = int(os.environ["TOTAL_CHUNKS"])
submit_dir = os.environ["SLURM_SUBMIT_DIR"]

experiment_dir = os.path.join(submit_dir, "experiments", experiment)
report_path = os.path.join(experiment_dir, "audit_report.json")

with open(report_path, "r") as f:
    report = json.load(f)

now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
summary = report["summary"]

# --- Determine per-step failures ---

# Step A: weights
weights_status = report["steps"]["weights"]["status"]
recover_a = weights_status != "OK"
reason_a = []
if recover_a:
    reason_a.append(f"weights: {weights_status}")

# Step B: matrices_task_*.zip
recover_b = False
recover_b_chunks = []
reason_b = []
for i, entry in enumerate(report["steps"]["matrices_zips"]):
    if entry["status"] != "OK":
        recover_b = True
        recover_b_chunks.append(str(i))
        reason_b.append(f"matrices_task_{i}.zip: {entry['status']}")

# Step C: adversarial_examples
recover_c = False
reason_c = []
for entry in report["steps"]["adversarial_examples"]:
    if entry["status"] != "OK":
        recover_c = True
        fname = os.path.basename(os.path.dirname(entry["path"]))
        reason_c.append(f"adversarial_examples/{fname}: {entry['status']}")

# Step D: adv_matrices_task_*.zip
recover_d = False
recover_d_chunks = []
reason_d = []
for i, entry in enumerate(report["steps"]["adv_matrices_zips"]):
    if entry["status"] != "OK":
        recover_d = True
        recover_d_chunks.append(str(i))
        reason_d.append(f"adv_matrices_task_{i}.zip: {entry['status']}")

# Step E: representation comparison
comparison_path = os.path.join(experiment_dir, "comparison", "representation_comparison.json")
recover_e = not os.path.exists(comparison_path)
reason_e = []
if recover_e:
    reason_e.append("comparison/representation_comparison.json missing")

# Step F: LaTeX tables (depends on E)
recover_f = recover_e
reason_f = []
if recover_f:
    reason_f.append("Propagated: depends on Step E")

# --- Propagate dependencies ---

# If A is bad, everything downstream needs re-run
if recover_a:
    if not recover_b:
        recover_b = True
        recover_b_chunks = [str(i) for i in range(total_chunks)]
        reason_b.append("Propagated: depends on Step A")
    if not recover_c:
        recover_c = True
        reason_c.append("Propagated: depends on Step A")

# If C is bad, D needs re-run
if recover_c:
    if not recover_d:
        recover_d = True
        recover_d_chunks = [str(i) for i in range(total_chunks)]
        reason_d.append("Propagated: depends on Step C")

# If A, B, C, or D changed, E needs re-run
if (recover_a or recover_b or recover_c or recover_d) and not recover_e:
    recover_e = True
    deps = []
    if recover_a: deps.append("Step A")
    if recover_b: deps.append("Step B")
    if recover_c: deps.append("Step C")
    if recover_d: deps.append("Step D")
    reason_e.append(f"Propagated: depends on {' + '.join(deps)}")

# If E changed, F needs re-run
if recover_e and not recover_f:
    recover_f = True
    reason_f.append("Propagated: depends on Step E")

recovery_needed = any([recover_a, recover_b, recover_c, recover_d,
                       recover_e, recover_f])

# --- Write recovery_plan.sh ---
plan_path = os.path.join(submit_dir, "experiments", experiment, "recovery_plan.sh")
os.makedirs(os.path.dirname(plan_path), exist_ok=True)

with open(plan_path, "w") as f:
    f.write("#!/bin/bash\n")
    f.write("# ==============================================================\n")
    f.write(f"# Recovery Plan for experiment: {experiment}\n")
    f.write(f"# Generated: {now}\n")
    f.write(f"# Audit summary: {summary}\n")
    f.write("# ==============================================================\n\n")
    f.write(f'RECOVERY_EXPERIMENT="{experiment}"\n')
    f.write(f'RECOVERY_TOTAL_CHUNKS={total_chunks}\n')
    f.write(f'RECOVERY_GENERATED_AT="{now}"\n\n')

    def write_step(f, name, label, recover, reasons, chunks=None):
        f.write(f"# --- Step {name}: {label} ---\n")
        f.write(f"RECOVER_STEP_{name}={'true' if recover else 'false'}\n")
        if chunks is not None:
            f.write(f'RECOVER_STEP_{name}_CHUNKS="{" ".join(chunks) if recover else ""}"\n')
        for r in reasons:
            f.write(f"#   {r}\n")
        f.write("\n")

    write_step(f, "A", "Training", recover_a, reason_a)
    write_step(f, "B", "Generate matrices", recover_b, reason_b, recover_b_chunks)
    write_step(f, "C", "Adversarial examples", recover_c, reason_c)
    write_step(f, "D", "Adversarial matrices", recover_d, reason_d, recover_d_chunks)
    write_step(f, "E", "Rep. Comparison", recover_e, reason_e)
    write_step(f, "F", "LaTeX Tables", recover_f, reason_f)

    f.write(f"RECOVERY_NEEDED={'true' if recovery_needed else 'false'}\n")

print(f"Recovery plan saved to: {plan_path}")
if recovery_needed:
    print("RECOVERY NEEDED — run job_recovery.sh to submit recovery jobs.")
else:
    print("No recovery needed — all artifacts OK.")
RECOVERY_PLAN_EOF

echo "Audit job complete."
