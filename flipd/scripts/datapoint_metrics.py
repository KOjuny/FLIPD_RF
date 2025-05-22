"""
This is a general-purpose script for ingesting a dataset, computing
a datapoint-wise metric across the dataset, and outputting
the metrics plus metadata to a CSV.
"""

import os
from pathlib import Path

import hydra
import mlflow
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

# get the environment variable IS_TESTING
# when testing, the main function will be called from a
# different file, so we need to import the tools module
# via the scripts package
if os.environ.get("IS_TESTING", False):
    from scripts import tools
else:
    import tools


@hydra.main(version_base=None, config_path="../conf/", config_name="metric")
@tools.MlflowDecorator(
    exclude_attributes=[  # The hydra attributes to remove from the mlflow logging
        "_all_data_transforms",
        "all_data_transforms",
    ],
    out_dir="./outputs",  # The directory where the artifacts are stored
    experiment_name="metric",  # The name of the experiment to be logged on mlflow
)
def main(cfg: DictConfig, artifact_dir: Path):

    # load scoring class and dataset
    metric = instantiate(cfg.metric)
    dataset = instantiate(cfg.metric_dataset)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, **cfg.dataloader)

    # Compute scores
    scores = metric.score_dataloader(dataloader)

    # Save scores
    if not cfg.dev_run:
        df = pd.DataFrame(scores, columns=[metric.__class__.__name__])
        if hasattr(dataset, "metadata"):
            df = pd.concat((df, dataset.metadata), axis="columns")
        df.to_csv(artifact_dir / "metrics.csv")


if __name__ == "__main__":

    tools.setup_root()
    main()
