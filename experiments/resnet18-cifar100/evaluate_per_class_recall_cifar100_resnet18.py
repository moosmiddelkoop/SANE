"""Per-class recall (acc_class_0..99) of every ResNet-18 in the CIFAR-100 zoo.

Evaluates checkpoint_000060 of each model on the CIFAR-100 test set (100 images
per class) and writes the result into the model's result.json, in the same way
as the unthi zoos: every line gets the keys acc_class_0..99, with the recall on
the line of training_iteration 60 and the sentinel -999 on all other lines.
The script only adds keys; removing the acc_class_* keys restores the original.
Re-running overwrites the values.

As a check, the mean recall (= accuracy, the test set is balanced) is compared
with the test_acc that the zoo logged at training_iteration 60.
"""

import json
import logging
import os
from pathlib import Path

import torch
from torchvision import datasets, transforms

from SANE.models.def_net import ResNet18
from SANE.utils import seed_everything

logging.basicConfig(level=logging.INFO)

SEED = 42
ZOO_ROOT = Path(
    "/projects/prjs2156/shared/wsl/cifar100_resnet18/cifar100_resnet18_kaiming_uniform_ep60_no_opt"
)
CIFAR100_ROOT = Path("/projects/prjs2156/shared/wsl/vision_datasets/CIFAR100")
EVAL_EPOCH = 60
NUM_CLASSES = 100
SENTINEL = -999.0
BATCH_SIZE = 2000

seed_everything(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# same normalization as data/vision_datasets/create_cifar100_dataset.py
test_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
    ]
)
testset = datasets.CIFAR100(root=CIFAR100_ROOT, train=False, transform=test_transforms)
images = torch.stack([image for image, _ in testset]).to(device)
labels = torch.tensor(testset.targets, device=device)
images_per_class = torch.bincount(labels, minlength=NUM_CLASSES)

model = ResNet18(channels_in=3, out_dim=NUM_CLASSES, init_type=None).to(device).eval()

model_dirs = sorted(d for d in ZOO_ROOT.iterdir() if d.is_dir())
logging.info(f"{len(model_dirs)} models in {ZOO_ROOT}")
accuracy_differences = []
for model_dir in model_dirs:
    checkpoint_path = model_dir / f"checkpoint_{EVAL_EPOCH:06d}" / "checkpoints"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    with torch.no_grad():
        predictions = torch.cat([model(batch).argmax(dim=1) for batch in images.split(BATCH_SIZE)])
    correct_per_class = torch.bincount(labels[predictions == labels], minlength=NUM_CLASSES)
    recall = (correct_per_class.double() / images_per_class).tolist()  # float64: 0.86, not 0.8600000143

    result_path = model_dir / "result.json"
    lines = [json.loads(line) for line in result_path.read_text().splitlines()]
    for line in lines:
        is_eval_line = line["training_iteration"] == EVAL_EPOCH
        for class_index in range(NUM_CLASSES):
            line[f"acc_class_{class_index}"] = recall[class_index] if is_eval_line else SENTINEL
    temporary_path = result_path.with_suffix(".json.tmp")
    temporary_path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    os.replace(temporary_path, result_path)

    logged_test_acc = next(l["test_acc"] for l in lines if l["training_iteration"] == EVAL_EPOCH)
    accuracy = sum(recall) / NUM_CLASSES
    accuracy_differences.append(abs(accuracy - logged_test_acc))
    logging.info(f"{model_dir.name}: acc {accuracy:.4f}, logged test_acc {logged_test_acc:.4f}")

logging.info(
    f"done: {len(model_dirs)} models; |acc - logged test_acc| "
    f"max {max(accuracy_differences):.4f}, mean {sum(accuracy_differences) / len(accuracy_differences):.4f}"
)
