from pathlib import Path
from urllib import request

import torch
import torch.nn.functional as F
from torchvision import transforms

from .base_metric import BaseSimMetric
from .utils import ImagesFromFilenames


class SSCDSimMetric(BaseSimMetric):
    def __init__(
        self,
        arch="resnet50",
        dist_threshold=0.55,
        is_ascend=False,
        index_type="inner_product",
        batch_size=50,
        num_workers=8,
    ):
        # init with hypers
        super().__init__(
            dist_threshold,
            is_ascend,
            index_type,
            batch_size,
        )

        # get model and transform
        self.model, self.transforms = self.get_model_and_transforms(arch)
        self.num_workers = num_workers

    def get_model_and_transforms(self, arch):
        # load the model (either from disc or from the internet)
        if arch == "resnet50":
            arch_path_name = "sscd_disc_mixup"
        elif arch == "resnet50_im":
            arch_path_name = "sscd_imagenet_mixup"
        elif arch == "resnet50_disc":
            arch_path_name = "sscd_disc_large"
        else:
            NotImplementedError("This model type does not exist for SSCD")

        model_url = (
            f"https://dl.fbaipublicfiles.com/sscd-copy-detection/{arch_path_name}.torchscript.pt"
        )
        model_save_path = (
            Path(__file__).parent / f"pretrained_models/{arch_path_name}.torchscript.pt"
        )
        model_save_path.parent.mkdir(exist_ok=True)

        try:
            model = torch.jit.load(model_save_path)
        except RuntimeError:
            request.urlretrieve(model_url, model_save_path)
            model = torch.jit.load(model_save_path)

        if torch.cuda.is_available():
            model = model.cuda()
        else:
            print("CUDA is not available. Using CPU instead.")
        model.eval()

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        test_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )

        return model, test_transform

    @torch.no_grad()
    def feature_transform(self, reference_list):
        """
        transform list of img path to the features that are used for memorization check
        """
        ref_dataset = ImagesFromFilenames(reference_list, transform=self.transforms)
        ref_dataloader = torch.utils.data.DataLoader(
            ref_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        # define model specific feature extraction
        features = []
        for samples in ref_dataloader:
            samples = samples.cuda()
            feats = self.model(samples).clone()
            feats = feats.cpu()
            features.append(feats)

        features = torch.cat(features, axis=0)
        features = F.normalize(features, dim=1, p=2)

        return features

    def compute_pairwise_distance(self, query):
        # search for closest reference samples first

        Dist, Idx = self.index.search(query, 1)  # D shape (query_bs, 1)

        closest_idx_list = Idx[:, 0]
        closest_dis_list = Dist[:, 0]
        closest_ref_list = [self.reference_list[closest_idx] for closest_idx in closest_idx_list]

        return closest_dis_list, closest_ref_list
