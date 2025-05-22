"""
This test is used primarily to check if the model design is
backwards compatible and that one can use the checkpoint URL
that is specified in the downloads directory.
"""

import os
from datetime import datetime
from pathlib import Path

import pytest

from scripts.datapoint_metrics import main as datapoint_metrics_main

from .utils import hydra_script_runner


@pytest.fixture
def default_overrides():
    """Returns a list of overrides that should happen for all of the configurations specified here"""
    current_datatime = datetime.now().strftime("%H:%M:%S_of_%y-%m-%d")  # add a timestamp
    return [
        "++mflow.experiment_name=hydra_test",
        "+mlflow.tags.timestamp=" + current_datatime,
        "+mlflow.tags.test_script=datapoint_metrics",
    ]


@pytest.fixture
def ground_truth_yaml_directory() -> Path:
    """Contains the directory containing the reference yaml files"""
    directory = Path(__file__).parent.parent / "resources" / "hydra_config" / "datapoint_metrics"
    os.makedirs(directory, exist_ok=True)
    return directory


@pytest.fixture
def generated_yaml_directory() -> Path:
    """All the generated yaml files will be dumped into this directory and then compared to the ground truth ones."""
    directory = (
        Path(__file__).parent.parent.parent / "outputs" / "hydra_config" / "datapoint_metrics"
    )
    os.makedirs(directory, exist_ok=True)
    return directory


# diffusion models for image data
lid_conditional_laion_aesthetics2k = [
    "metric=lid_conditional",
    "metric_dataset=laion_aesthetics2k",
    "metric_dataset.raw_dataset.subset_size=128",
]

# A list of all possible settings to test:
all_settings = [
    (0, "dev_lid_conditional_laion_aesthetics2k", lid_conditional_laion_aesthetics2k),
]


@pytest.mark.parametrize(
    "setting",
    all_settings,
)
@pytest.mark.parametrize(
    "dummy",
    [True, False],
)
def test_checkpoints_scripts(
    default_overrides,
    setting,
    ground_truth_yaml_directory,
    generated_yaml_directory,
    dummy,
):
    level, setting_name, new_overrides = setting
    overrides = (default_overrides or []) + (new_overrides or [])
    overrides += ["++mlflow.tags.setting=" + setting_name]
    overrides += ["++mlflow.experiment_name=hydra_tests"]

    hydra_script_runner(
        script_level=level,
        setting_name=setting_name,
        overrides=overrides,
        ground_truth_yaml_directory=ground_truth_yaml_directory,
        generated_yaml_directory=generated_yaml_directory,
        dummy=dummy,
        main_fn=datapoint_metrics_main,
        script_name="metric",
        exclude_attributes=[
            "mlflow.tags.timestamp",  # exclude the timestamp from the mlflow tags when comparing configurations
        ],
    )
