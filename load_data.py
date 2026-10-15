import sys
from pathlib import Path

# Choose the dataset.
dataset_name = "Store_Sales"

root_dir = Path(__file__).resolve().parent
dataset_dir = root_dir / dataset_name

# Use dataloader.py of the root directory, and dataset.py of the selected dataset.
sys.path[:0] = [str(root_dir), str(dataset_dir)]

from dataloader import (
    read_all_configs,
    MultiModalDataset,
    get_multi_modal_datalset,
)

configs = read_all_configs(str(dataset_dir / "dataset.yaml"))
dataset = MultiModalDataset(configs, dataset_dir)

# Choose the task.
task_name = "store_type_classification"
task_config = dataset_dir / "tasks" / task_name / "task.yaml"

train_ds, valid_ds, test_ds = get_multi_modal_datalset(
    dataset,
    task_config=task_config
)