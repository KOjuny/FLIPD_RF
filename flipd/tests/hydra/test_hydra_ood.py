"""
This test is used primarily to check if the model design is
backwards compatible and that one can use the checkpoint URL
that is specified in the downloads directory.
"""

import os
from datetime import datetime
from pathlib import Path

import pytest

from scripts.download_resources import main as download_checkpoints_main
from scripts.train import main as train_main

from .utils import hydra_script_runner


@pytest.fixture
def default_overrides():
    """Returns a list of overrides that should happen for all of the configurations specified here"""
    current_datatime = datetime.now().strftime("%H:%M:%S_of_%y-%m-%d")  # add a timestamp
    return [
        "++mflow.experiment_name=hydra_test",
        "+mlflow.tags.timestamp=" + current_datatime,
        "+mlflow.tags.test_script=train",
        "+mlflow.tags.test_type=ood",
    ]


@pytest.fixture
def ground_truth_yaml_directory() -> Path:
    """Contains the directory containing the reference yaml files"""
    directory = Path(__file__).parent.parent / "resources" / "hydra_config" / "ood"
    os.makedirs(directory, exist_ok=True)
    return directory


@pytest.fixture
def generated_yaml_directory() -> Path:
    """All the generated yaml files will be dumped into this directory and then compared to the ground truth ones."""
    directory = Path(__file__).parent.parent.parent / "outputs" / "hydra_config" / "ood"
    os.makedirs(directory, exist_ok=True)
    return directory


flow_likelihood_paradox_fmnist_mnist = [
    "dataset=fmnist",
    "+ood_dataset=mnist",
    "+experiment=train_flow_greyscale",
    "+checkpoint=flow_fmnist",
    "+ood=flow_likelihood_paradox",
    "ood_subsample_size=2",
    "ood_batch_size=2",
]


flow_likelihood_paradox_mnist_fmnist = [
    "dataset=fmnist",
    "+ood_dataset=mnist",
    "+experiment=train_flow_greyscale",
    "+checkpoint=flow_fmnist",
    "+ood=flow_likelihood_paradox",
    "ood_subsample_size=2",
    "ood_batch_size=2",
]

diffusion_likelihood_paradox_fmnist_mnist = [
    "dataset=fmnist",
    "+ood_dataset=mnist",
    "+experiment=train_diffusion_greyscale",
    "+checkpoint=diffusion_fmnist",
    "+ood=diffusion_likelihood_paradox",
    "ood_subsample_size=2",
    "ood_batch_size=2",
]

diffusion_likelihood_paradox_fmnist_mnist = [
    "dataset=mnist",
    "+ood_dataset=fmnist",
    "+experiment=train_diffusion_greyscale",
    "+checkpoint=diffusion_mnist",
    "+ood=diffusion_likelihood_paradox",
    "ood_subsample_size=2",
    "ood_batch_size=2",
]

fmnist_vs_mnist_flow_lid_curve = [
    "dataset=fmnist",
    "+ood_dataset=mnist",
    "+experiment=train_flow_greyscale",
    "+checkpoint=flow_fmnist",
    "+ood=flow_lid_curve",
    "ood_subsample_size=2",
    "ood_batch_size=2",
]

mnist_vs_fmnist_flow_lid_curve = [
    "dataset=mnist",
    "+ood_dataset=fmnist",
    "+experiment=train_flow_greyscale",
    "+checkpoint=flow_mnist",
    "+ood=flow_lid_curve",
    "ood_subsample_size=2",
    "ood_batch_size=2",
]

fmnist_vs_mnist_diffusion_nb_curve = [
    "dataset=fmnist",
    "+ood_dataset=mnist",
    "+experiment=train_diffusion_greyscale",
    "+checkpoint=diffusion_fmnist",
    "+ood=diffusion_lid_curve",
    "ood_subsample_size=2",
    "ood_batch_size=2",
]

mnist_vs_fmnist_diffusion_nb_curve = [
    "dataset=mnist",
    "+ood_dataset=fmnist",
    "+experiment=train_diffusion_greyscale",
    "+checkpoint=diffusion_mnist",
    "+ood=diffusion_lid_curve",
    "ood_subsample_size=2",
    "ood_batch_size=2",
]

fmnist_vs_mnist_dual_likelihood_lid_flow = [
    "dataset=fmnist",
    "+ood_dataset=mnist",
    "+experiment=train_flow_greyscale",
    "+checkpoint=flow_fmnist",
    "+ood=flow_lid_likelihood",
    "+lid_metric.singular_value_threshold=-3",
    "ood_subsample_size=2",
    "ood_batch_size=2",
]

mnist_vs_fmnist_dual_likelihood_lid_flow = [
    "dataset=mnist",
    "+ood_dataset=fmnist",
    "+experiment=train_flow_greyscale",
    "+checkpoint=flow_mnist",
    "+ood=flow_lid_likelihood",
    "+lid_metric.singular_value_threshold=-3",
    "ood_subsample_size=2",
    "ood_batch_size=2",
]

fmnist_vs_mnist_dual_likelihood_lid_diffusion = [
    "dataset=fmnist",
    "+ood_dataset=mnist",
    "+experiment=train_diffusion_greyscale",
    "+checkpoint=diffusion_fmnist",
    "+ood=diffusion_lid_likelihood",
    "+lid_metric.singular_value_threshold=-3",
    "ood_subsample_size=2",
    "ood_batch_size=2",
]

mnist_vs_fmnist_dual_likelihood_lid_diffusion = [
    "dataset=mnist",
    "+ood_dataset=fmnist",
    "+experiment=train_diffusion_greyscale",
    "+checkpoint=diffusion_mnist",
    "+ood=diffusion_lid_likelihood",
    "+lid_metric.singular_value_threshold=-3",
    "ood_subsample_size=2",
    "ood_batch_size=2",
]

all_settings = [
    (0, "flow_likelihood_paradox_fmnist_mnist", flow_likelihood_paradox_fmnist_mnist),
    (1, "flow_likelihood_paradox_mnist_fmnist", flow_likelihood_paradox_mnist_fmnist),
    (2, "diffusion_likelihood_paradox_fmnist_mnist", diffusion_likelihood_paradox_fmnist_mnist),
    (3, "diffusion_likelihood_paradox_fmnist_mnist", diffusion_likelihood_paradox_fmnist_mnist),
    (4, "fmnist_vs_mnist_flow_lid_curve", fmnist_vs_mnist_flow_lid_curve),
    (5, "mnist_vs_fmnist_flow_lid_curve", mnist_vs_fmnist_flow_lid_curve),
    (6, "fmnist_vs_mnist_diffusion_nb_curve", fmnist_vs_mnist_diffusion_nb_curve),
    (7, "mnist_vs_fmnist_diffusion_nb_curve", mnist_vs_fmnist_diffusion_nb_curve),
    (8, "fmnist_vs_mnist_dual_likelihood_lid_flow", fmnist_vs_mnist_dual_likelihood_lid_flow),
    (9, "mnist_vs_fmnist_dual_likelihood_lid_flow", mnist_vs_fmnist_dual_likelihood_lid_flow),
    (
        10,
        "fmnist_vs_mnist_dual_likelihood_lid_diffusion",
        fmnist_vs_mnist_dual_likelihood_lid_diffusion,
    ),
    (
        11,
        "mnist_vs_fmnist_dual_likelihood_lid_diffusion",
        mnist_vs_fmnist_dual_likelihood_lid_diffusion,
    ),
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
    is_cicd = os.environ.get("CICD")
    if is_cicd:
        pytest.skip(f"Skipping due to not having checkpoints in CI/CD")
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
        main_fn=train_main,
        script_name="train",
        exclude_attributes=[
            "mlflow.tags.timestamp",  # exclude the timestamp from the mlflow tags when comparing configurations
            "train.ckpt_path",  # this is a private info
        ],
    )
