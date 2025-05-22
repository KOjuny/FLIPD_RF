import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.utils
import torchvision.transforms.functional as TVF
from lightning import Trainer
from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm

from data.datasets import LIDDataset
from models.training import LightningDGM, LightningEnsemble


class MonitorMetrics(Callback, ABC):
    """
    This is a generic callback used for monitoring the characteristics of certain datapoints w.r.t. to the
    generative model at hand. All of these generative models should be either a LightningDGM or a LightningEnsemble.
    This is a pattern we use across this codebase to implement many callbacks. For example, LID monitoring is an example
    of this callback, where for each datapoint, we are interested in LID-related metrics of that datapoint.

    Thus, if you want to monitor metrics associated with each datapoint on your generative model, it is recommended
    to inherit this callback and implement the following:

    1. `_on_train_start`:
        This function is called at the start of the training loop and is used for any form of initalization.
        For example, if you are implementing a monitoring on the LID estimators, you may want to instantiate the
        LID estimator here.


    2. `_compute_logs`:
        Stores a metric for a batch of data. This function returns a dictionary with the following scheme:
        {
            metric1: torch.Tensor of shape (batch_size, ...),
            metric2: torch.Tensor of shape (batch_size, ...),
            ...
        }
        and for each datapoint (i) in the batch, the value of the metric is stored in the ith row of the tensor.

    3. `_log`:
        After calling `_compute_logs` on the entire dataset (an amalgamation of all the batches), this function is called
        to log the metrics in whatever format you desire. For example, you may want to log the metrics in a csv file or
        in a plot. This function takes in a dictionary, that is now of the following form:
        {
            metric1: torch.Tensor of shape (total_data_count, ...),
            metric2: torch.Tensor of shape (total_data_count, ...),
            ...
        }
        As you can see, the data is automatically concatenated across all the batches.

    Note that this callback does not shuffle the data, so every datapoint has a unique and global identifier.

    That being said, this callback logs the information of the data for you by default. You can override
    the name of the callback as {callback_name} and you will automatcally see the following files on your mlflow
    artifacts directory:

    1. {callback_name}/manifold_info.csv: (only relevant for LIDDatasets)
        This is a csv file containing information of the data manifold (if available)
        it has a row count equals to the number of subsamples under consideration.
        It contains three columns, one being the index of the data (this index is a global
        identifier of datapoints), the second column is the submanifold index that it
        is associated with and the third column is the true LID. The information of the
        manifold is only available when dataset is a LIDDataset, otherwise, the submanifold
        will be assigned to 0 and the true LID will be assigned to -1.


    2. {callback_name}/samples/datapoints.csv and {callback_name}/samples/datapoints_transformed.csv:
        **Only when the data is tabular**, meaning that when the `save_image` flag is set to False
        this file is logged which contains the actual information of the subsampled datapoints.
        The row count is equal to the number of subsamples under consideration and the column count
        is equal to the number of ambient dimensions (lollipop will have 2 columns).
        The first csv file will not do the sampling_transform on the data and the second csv file
        will do the sampling_transform on the data.

    3. {callback_name}/samples/idx={data_idx:07d}.npy and {callback_name}/samples/idx={data_idx:07d}_transformed.npy:
        **Only when the data is image**, meaning that when the `save_image` flag is set to True
        all the subsampled datapoints are stored both as numpy files and as image files. The scheme
        of the `data_idx` also follows the same scheme as the csv files.

    **Logging Hyperparameters**:

    1. frequency:
        The frequency at which to log the metrics.
    2. subsample_size:
        The number of subsamples to consider from the base dataset. This is used to make things more feasible and faster
        and it is based on the premise that subsampling data from the original dataset and monitoring the metrics on it
        is a good approximation of the entire dataset.
    3. verbose:
        Whether to print the progress of the monitoring process.
    4. batch_size:
        Data is processed in batches in monitoring, this is the batch size to use. You can tune based on the resources
        you have available.
    """

    # All the callbacks that involve subsampling from the original dataset should be seeded with this value.
    # otherwise, the results while monitoring the model will be inconsistent across callbacks.

    SUBSAMPLING_SEED: int = 42

    def __init__(
        self,
        dataset: LIDDataset | TorchDataset,
        device: torch.device,
        frequency: int = 1,
        subsample_size: int | None = None,
        batch_size: int = 128,
        verbose: bool = True,
        save_image: bool = False,
    ):
        """
        Args:
            dataset (LIDDataset | TorchDataset):
                This is the dataset to monitor metrics on, it is first subsampled and then iterated upon.
                This dataset is always a torch dataset. If it is defined as an LIDDataset, the manifold
                information of it will also be stored within a separate csv file.

            device (torch.device):
                The device to use for computing metrics in this callback.

            frequency (int | None):
                The frequency at which to log the metrics. By default, it is set to one, which means
                it will log everything after every single epoch.

            subsample_size (int | None): The number of subsamples to consider from the original dataset.

            batch_size (int):
                This is the batch size being used to iterate over the data. The data is by default
                loaded into a torch dataset and shuffled (using a seed) and then the batch_size for
                the dataloader is specified here.

                NOTE: it is required that the subsample_size is divisible by the batch_size.

            verbose (bool):
                Whether to print the progress of the callback or not.

            save_image (bool):
                Whether to save the image of the subsampled data, only set to true if the data is image.
        """
        # flag if data is LID dataset
        self.is_lid_dataset = isinstance(dataset, LIDDataset)
        assert isinstance(
            dataset, TorchDataset
        ), f"Invalid dataset {dataset}, should be a torch dataset."

        # set frequency
        self.frequency = frequency
        self.rem = self.frequency

        # Datasets
        self.dataset = dataset
        self.batch_size = batch_size
        self.dloader = TorchDataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )  # set shuffle true but control with seed!
        self.call_cnt = 0

        self.verbose = verbose
        self.device = device
        if subsample_size is None:
            self.subsample_size = (len(dataset) // batch_size) * batch_size  # ensure divisibility
        else:
            self.subsample_size = min(len(dataset), subsample_size)

        # flag of data modality
        self.save_image = save_image

        self.tqdm_batch_count = (self.subsample_size + batch_size - 1) // batch_size

        assert subsample_size % batch_size == 0, "Subsample size should be divisible by batch size."

    def _check_pl_module(self, pl_module: LightningDGM | LightningEnsemble):
        if isinstance(pl_module, LightningEnsemble):
            for i, lightning_dgm in enumerate(pl_module.lightning_dgms):
                assert hasattr(
                    lightning_dgm, "dgm"
                ), f"Ensemble model {i} with type {type(lightning_dgm)} does not have a 'dgm' attribute"
        else:
            assert hasattr(
                pl_module, "dgm"
            ), f"Model attribute 'dgm' not found in the lightning module of instance {type(pl_module)}"

    def _init_data_saving(self, pl_module: LightningDGM):
        """
        Save the dataset information, this includes saving all the samples
        that are being considered, alongside the manifold information.

        This is all done in a reproducible fashion by setting the seed.
        """
        # store the manifold information here
        all_lid = []
        all_idx = []

        # store both the original and transformed data
        all_datapoints = []
        all_datapoints_transformed = []

        with torch.random.fork_rng():
            # seed for fixing the dataloader iterations
            torch.random.manual_seed(MonitorMetrics.SUBSAMPLING_SEED)
            current_idx = 0
            my_data_iterator = self.dloader
            if self.verbose:
                my_data_iterator = tqdm(
                    my_data_iterator,
                    total=self.tqdm_batch_count,
                    desc=f"[Monitoring Callback {self.callback_name}] Initial iteration through data ...",
                )
            for loader_batch in my_data_iterator:
                # load the data and extract the submanifold idx, lid, and the underlying data itself
                # set the LID to -1 for the non-LID dataset and the idx to 0.
                if self.is_lid_dataset:
                    data_batch, lid_batch, idx_batch = loader_batch
                else:
                    data_batch = loader_batch
                    lid_batch = -1 * torch.ones(data_batch.shape[0], device=data_batch.device)
                    idx_batch = torch.zeros(data_batch.shape[0], device=data_batch.device).long()
                assert isinstance(data_batch, torch.Tensor), "Invalid data batch"
                if current_idx >= self.subsample_size:
                    break

                # append the data to the list
                all_lid.append(lid_batch.cpu())
                all_idx.append(idx_batch.cpu())
                all_datapoints.append(data_batch.cpu())
                if pl_module.sampling_transform is not None:
                    data_batch_transformed = pl_module.sampling_transform(data_batch)
                else:
                    data_batch_transformed = data_batch
                all_datapoints_transformed.append(data_batch_transformed)

                # Now iterate over every single datapoint in the batch and store it if the save_image flag is set
                if self.save_image:
                    for single_datapoint, single_datapoint_transformed in zip(
                        data_batch, data_batch_transformed
                    ):
                        np.save(
                            self.artifact_dir
                            / self.samples_without_transform_fstr.format(call_cnt=current_idx),
                            single_datapoint.cpu().numpy(),
                        )
                        np.save(
                            self.artifact_dir
                            / self.samples_with_transform_fstr.format(call_cnt=current_idx),
                            single_datapoint_transformed.cpu().numpy(),
                        )
                        assert single_datapoint.dim() in [
                            2,
                            3,
                        ], "Invalid image shape for storing, set the save_image flag appropriately if you are dealing with tabular data!"

                        transformed_pil = TVF.to_pil_image(single_datapoint_transformed)
                        normal_pil = TVF.to_pil_image(single_datapoint)

                        mlflow.log_image(
                            transformed_pil,
                            self.samples_with_transform_image_fstr.format(call_cnt=current_idx),
                        )
                        mlflow.log_image(
                            normal_pil,
                            self.samples_without_transform_image_fstr.format(call_cnt=current_idx),
                        )
                        current_idx += 1
                else:
                    current_idx += self.batch_size

        if not self.save_image:
            all_datapoints = torch.flatten(torch.cat(all_datapoints, dim=0), start_dim=1).numpy()
            all_datapoints_transformed = torch.flatten(torch.cat(all_datapoints_transformed, dim=0), start_dim=1).numpy()
            df = pd.DataFrame(all_datapoints)
            df.to_csv(
                self.artifact_dir / self.callback_name / "samples" / "datapoints.csv",
                index=True,
            )
            df = pd.DataFrame(all_datapoints_transformed)
            df.to_csv(
                self.artifact_dir / self.callback_name / "samples" / "datapoints_transformed.csv",
                index=True,
            )

        if self.verbose:
            print(f"[Monitoring Callback {self.callback_name}] Storing manifold information ...")

        pd.DataFrame(
            {
                "lid": torch.cat(all_lid).cpu().numpy(),
                "submanifold": torch.cat(all_idx).cpu().numpy(),
            }
        ).to_csv(self.artifact_dir / self.path_data_manifold_info, index=True)

    @abstractmethod
    def _on_train_start(self, trainer: Trainer, pl_module: LightningDGM | LightningEnsemble):
        raise NotImplementedError("This method should be implemented by the subclass.")

    @property
    @abstractmethod
    def callback_name(self):
        raise NotImplementedError("This method should be implemented by the subclass.")

    def on_train_start(self, trainer: Trainer, pl_module: LightningDGM | LightningEnsemble) -> None:

        # (1) instantiate and setup
        self._check_pl_module(pl_module=pl_module)

        # (2) setup the artifact directory
        self.artifact_dir: Path = Path(trainer.default_root_dir)

        # (3) setup the callback
        self._on_train_start(trainer=trainer, pl_module=pl_module)

        # (4) take care of all the paths required for logging in this callback
        self.samples_without_transform_fstr = self.callback_name + "/samples/idx={call_cnt:07d}.npy"
        self.samples_with_transform_fstr = (
            self.callback_name + "/samples/idx={call_cnt:07d}_transformed.npy"
        )
        self.samples_with_transform_image_fstr = (
            self.callback_name + "/samples/idx={call_cnt:07d}_transformed.png"
        )
        self.samples_without_transform_image_fstr = (
            self.callback_name + "/samples/idx={call_cnt:07d}.png"
        )
        self.path_data_manifold_info = self.callback_name + "/manifold_info.csv"
        os.makedirs(self.artifact_dir / self.callback_name / "samples", exist_ok=True)

        # (5) store everything related to the data that is being subsampled
        self._init_data_saving(pl_module=pl_module)

    @abstractmethod
    def _compute_logs(
        self,
        batch: torch.Tensor,
        trainer: Trainer,
        pl_module: LightningDGM | LightningEnsemble,
        iterator,
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("This method should be implemented by the subclass.")

    @abstractmethod
    def _log(
        self,
        logging_results: Dict[str, torch.Tensor],
        trainer: Trainer,
        pl_module: LightningDGM | LightningEnsemble,
    ):
        raise NotImplementedError("This method should be implemented by the subclass.")

    @torch.no_grad()
    def on_train_epoch_end(
        self, trainer: Trainer, pl_module: LightningDGM | LightningEnsemble
    ) -> None:
        # only log whenever the frequency is reached
        self.rem -= 1
        self.call_cnt += 1
        if self.rem != 0:
            return
        self.rem = self.frequency

        # Store all the logging results for all the datapoints
        # the scheme:
        # logging_results[key] = [value(1), value(2), ..., value_(subsample_size)] of shape (subsample_size, ...)
        logging_results = None
        with torch.random.fork_rng():
            torch.random.manual_seed(MonitorMetrics.SUBSAMPLING_SEED)
            current_idx = 0
            my_iterator = self.dloader
            if self.verbose:
                my_iterator = tqdm(
                    my_iterator,
                    total=self.tqdm_batch_count,
                    desc=f"[Monitoring Callback {self.callback_name}] Computing metrics for batches ...",
                )
            for loader_batch in my_iterator:

                if current_idx < self.subsample_size:
                    new_results = self._compute_logs(
                        batch=loader_batch,
                        trainer=trainer,
                        pl_module=pl_module,
                        iterator=my_iterator,
                    )
                    if logging_results is None:
                        logging_results = {key: [] for key in new_results.keys()}
                    for key, content in new_results.items():
                        assert isinstance(
                            content, torch.Tensor
                        ), f"All the results from _compute_logs should be a dictionary of {{keys: torch.tensor}}, but got {type(content)} instead."

                        logging_results[key].append(content.cpu())

                else:
                    break
                current_idx += self.batch_size

        # concatenate everything
        for key in logging_results.keys():
            logging_results[key] = torch.cat(logging_results[key], dim=0)

        self._log(
            logging_results=logging_results,
            trainer=trainer,
            pl_module=pl_module,
        )
