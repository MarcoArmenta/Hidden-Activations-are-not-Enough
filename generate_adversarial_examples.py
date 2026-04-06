import sys
import torch
import torchattacks
from torch.utils.data import TensorDataset, DataLoader
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Union

from utils.utils import get_model, get_dataset, subset, get_num_classes, get_input_shape, get_device
from constants.constants import DEFAULT_EXPERIMENTS, ATTACKS
from utils.atomic_io import atomic_torch_save


def parse_args(
        parser:Union[ArgumentParser, None] = None
    ) -> Namespace:
    """
        Args:
            parser: the parser to use.
        Returns:
            The parsed arguments.
    """
    if parser is None:
        parser = ArgumentParser()
    parser.add_argument(
        "--experiment_name", "--experiment",
        type = str,
        default = 'alexnet_cifar10',
        dest = "experiment_name",
        help = "Name of experiment <<network>>_<<dataset>>"
    )
    parser.add_argument(
        "--test_size",
        type = int,
        default = -1,
        help = "Size of subset of test data from where to generate adversarial examples."
              "As default -1 takes 10k test samples"
    )
    parser.add_argument(
        "--temp_dir",
        type = str,
        help = "Temporary directory for reading data when using clusters."
    )
    parser.add_argument(
        "--attacks",
        nargs = "+",
        default = None,
        help = "Subset of attacks to run (e.g. --attacks FGSM PGD CW). "
               "If not specified, all attacks from ATTACKS are run."
    )
    parser.add_argument(
        "--no_auto_test",
        action = "store_true",
        default = False,
        help = "Skip automatic prepending of 'test' (VANILA) attack. "
               "Use when running individual attacks in parallel Slurm jobs."
    )
    return parser.parse_args()


def apply_attack(
        attack_name: str,
        data: torch.Tensor,
        labels: torch.Tensor,
        weights_path: Path,
        architecture_index: int,
        path_adv_examples: Path,
        input_shape,
        num_classes: int,
        batch_size: int = 8,
    ):
    device = get_device()

    attack_save_path = path_adv_examples / f'{attack_name}/adversarial_examples.pth'
    wrong_pred_save_path = path_adv_examples / f'{attack_name}/wrong_predictions.pth'
    attack_save_path.parent.mkdir(parents=True, exist_ok=True)

    if attack_save_path.exists():
        print(f"Attack {attack_name} exists.")
        return

    print(f"Attacking with {attack_name}", flush=True)
    model = get_model(
        path = weights_path,
        architecture_index = architecture_index,
        input_shape = input_shape,
        num_classes = num_classes,
        device = device
    )
    model.eval()

    # don't move the whole dataset to device (that causes OOM)
    # data = data.to(device)
    # labels = labels.to(device)

    # prepare DataLoader to iterate in small batches
    ds = TensorDataset(data, labels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=True)

    # Build attack instance lazily (handles missing attacks in different torchattacks versions)
    attack_map = {
        "test": "VANILA", "GN": "GN", "FGSM": "FGSM", "PGD": "PGD",
        "EOTPGD": "EOTPGD", "MIFGSM": "MIFGSM", "VMIFGSM": "VMIFGSM",
        "CW": "CW", "DeepFool": "DeepFool", "Pixle": "Pixle",
        "APGD": "APGD", "APGDT": "APGDT", "FAB": "FAB", "Square": "Square",
        "SPSA": "SPSA", "EADL1": "EADL1", "EADEN": "EADEN",
    }
    attack_cls_name = attack_map.get(attack_name)
    if attack_cls_name is None:
        print(f"Unknown attack {attack_name}")
        return
    attack_cls = getattr(torchattacks, attack_cls_name, None)
    if attack_cls is None:
        print(f"WARNING: Attack {attack_name} ({attack_cls_name}) not available in torchattacks {torchattacks.__version__}. Skipping.", flush=True)
        return
    attack_instance = attack_cls(model)

    if attack_name == "test":
        # run on entire dataset in batches but save everything
        adv_list = []
        labels_list = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            with torch.no_grad():
                attacked = attack_instance(xb, yb)
            adv_list.append(attacked.cpu())
            labels_list.append(yb.cpu())
            del xb, yb, attacked
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        atomic_torch_save(torch.cat(adv_list), attack_save_path)
        atomic_torch_save(torch.cat(labels_list), path_adv_examples / f'{attack_name}/labels.pth')
        del adv_list, labels_list
        return

    # For real attacks: store only misclassified adversarial examples to save RAM
    adv_saved = []
    wrong_preds_saved = []
    total = 0
    misclassified = 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        try:
            attacked_batch = attack_instance(xb, yb)  # most attacks operate batchwise
        except Exception as e:
            print(f"Error applying attack {attack_name} on a batch: {e}")
            # free and continue to next batch / or break depending on severity
            del xb, yb
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        with torch.no_grad():
            preds = torch.argmax(model(attacked_batch), dim=1)

        mis_idx = (yb != preds)
        miscount_batch = mis_idx.sum().item()
        misclassified += miscount_batch
        total += xb.size(0)

        if miscount_batch > 0:
            adv_saved.append(attacked_batch[mis_idx].cpu())
            wrong_preds_saved.append(preds[mis_idx].cpu())

        # free GPU memory from this batch
        del xb, yb, attacked_batch, preds, mis_idx
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Attack: {attack_name}. Misclassified after attack: {misclassified} out of {total}.", flush=True)

    if len(adv_saved) > 0:
        atomic_torch_save(torch.cat(adv_saved), attack_save_path)
        atomic_torch_save(torch.cat(wrong_preds_saved), wrong_pred_save_path)
    else:
        print(f"  WARNING: {attack_name} produced 0 misclassified examples. Skipping save.", flush=True)
        # Write a zero-results marker so downstream can distinguish from crash
        save_dir = path_adv_examples / f'{attack_name}'
        save_dir.mkdir(parents=True, exist_ok=True)
        marker_path = save_dir / 'zero_misclassifications.txt'
        with open(marker_path, 'w') as f:
            f.write(f"Attack {attack_name} produced 0 misclassified adversarial examples\n")

    # cleanup
    del adv_saved, wrong_preds_saved, model, attack_instance
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate_adversarial_examples(
        exp_dataset_test: torch.Tensor,
        exp_labels_test: torch.Tensor,
        weights_path: Path,
        architecture_index: int,
        experiment_name: str,
        input_shape,
        num_classes: int,
        attacks: list = None,
        no_auto_test: bool = False,
    ) -> None:
    """
        Args:
            exp_dataset_test: the test set.
            exp_labels_test: the labels of the test set.
            weights_path: the path to the weights.
            architecture_index: the index of the architecture (See constants/constants.py).
            experiment_name: the name of the experiment (See constants/constants.py).
            input_shape: the shape of the input.
            num_classes: the number of classes.
            attacks: optional list of attack names to run. If None, all ATTACKS are used.
            no_auto_test: if True, skip automatic prepending of 'test' attack.
    """

    experiment_dir = Path(f'experiments/{experiment_name}/adversarial_examples')
    experiment_dir.mkdir(parents=True, exist_ok=True)

    print("Generating adversarial examples...", flush=True)

    exp_dataset_test = exp_dataset_test.detach().clone()
    exp_labels_test = exp_labels_test.detach().clone()

    attack_list = attacks if attacks is not None else ATTACKS
    run_list = attack_list if no_auto_test else ["test"] + attack_list
    failed_attacks = 0
    for attack_name in run_list:
        try:
            apply_attack(attack_name,
                         exp_dataset_test,
                         exp_labels_test,
                         weights_path,
                         architecture_index,
                         experiment_dir,
                         input_shape,
                         num_classes)
        except Exception as e:
            print(f'ERROR: Attack {attack_name} failed entirely: {type(e).__name__}: {e}', flush=True)
            failed_attacks += 1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

    total_attacks = len(run_list)
    if failed_attacks == total_attacks:
        print(f"FATAL: All {total_attacks} attacks failed.", flush=True)
        sys.exit(1)
    elif failed_attacks > 0:
        print(f"WARNING: {failed_attacks}/{total_attacks} attacks failed.", flush=True)


def main() -> None:
    """
        Main function to generate adversarial examples.
    """
    args = parse_args()
    if args.experiment_name is None:
        raise ValueError("Default index not specified in constants/constants.py")

    experiment = args.experiment_name
    architecture_index = DEFAULT_EXPERIMENTS[experiment]['architecture_index']
    dataset = DEFAULT_EXPERIMENTS[experiment]['dataset']
    epoch = DEFAULT_EXPERIMENTS[experiment]['epochs']

    print("Experiment: ", experiment)

    if args.temp_dir is not None:
        weights_path = Path(f'{args.temp_dir}/experiments/{experiment}/weights/epoch_{epoch}.pth')
    else:
        weights_path = Path(f'experiments/{experiment}/weights/epoch_{epoch}.pth')

    if not weights_path.exists():
        raise ValueError(f"Couldn't find weights at {weights_path}")

    input_shape = get_input_shape(dataset)
    num_classes = get_num_classes(dataset)
    _, test_set = get_dataset(dataset, data_loader=False, data_path=args.temp_dir)
    test_size = len(test_set) if args.test_size == -1 else args.test_size
    exp_dataset_test, exp_labels_test = subset(test_set, test_size, input_shape=input_shape)

    # Quick accuracy check before running attacks
    device = get_device()
    model = get_model(
        path=weights_path,
        architecture_index=architecture_index,
        input_shape=input_shape,
        num_classes=num_classes,
        device=device,
    )
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_data, batch_labels in DataLoader(
            TensorDataset(exp_dataset_test, exp_labels_test), batch_size=64, shuffle=False
        ):
            if total >= 500:
                break
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            outputs = model(batch_data)
            _, predicted = torch.max(outputs, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
    accuracy = correct / total if total > 0 else 0
    print(f"  Model accuracy on {total} test samples: {accuracy:.4f}", flush=True)
    if accuracy < 0.1:
        print(f"  WARNING: Model accuracy is very low ({accuracy:.4f}). "
              f"Adversarial examples may not be meaningful.", flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    generate_adversarial_examples(
        exp_dataset_test = exp_dataset_test,
        exp_labels_test = exp_labels_test,
        weights_path = weights_path,
        architecture_index = architecture_index,
        experiment_name = experiment,
        input_shape = input_shape,
        num_classes = num_classes,
        attacks = args.attacks,
        no_auto_test = args.no_auto_test,
    )


if __name__ == "__main__":
    main()
