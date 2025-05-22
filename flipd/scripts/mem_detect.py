from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import sys

sys.path.append("./")
from metrics.mem.metrics.utils import get_image_dataset_in_dir


@hydra.main(version_base=None, config_path="../conf/", config_name="mem_check")
def main(cfg: DictConfig):
    # initialize similarity estimator
    sim_estimator = instantiate(cfg.sim_metric)

    # Get reference data
    # trainset_list (img path list) and trainset_labels (not used here)
    trainset_list, _ = get_image_dataset_in_dir(cfg.refset_path)

    # build reference database
    sim_estimator.build_reference_database(trainset_list)

    # Get query data
    # queryset_list (img path list) and queryset_labels (not used here)
    queryset_list, _ = get_image_dataset_in_dir(cfg.queryset_path)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    distance_save_path = out_dir / f"distances_{cfg.metric_name}.pkl"
    grid_save_path = out_dir / f"match_grid_{cfg.metric_name}.png"

    # Get distances (sorted)
    sorted_results = sim_estimator.get_ranked_distances(
        queryset_list, distance_save_path=distance_save_path
    )

    # filter memorization with  threshold
    filtered_results = sim_estimator.filter_memorization_with_threshold(sorted_results)

    # save results
    sim_estimator.visualize(filtered_results, save_path=grid_save_path)


if __name__ == "__main__":
    from tools import setup_root

    setup_root()
    main()
