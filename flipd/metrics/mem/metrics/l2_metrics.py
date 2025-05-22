import numpy as np
from PIL import Image

from .base_metric import BaseSimMetric


def get_img_pixel_input(img_list):
    """load all training images and flatten to (N, ndim) array"""
    img_inputs = np.array(
        [np.array(Image.open(img)).reshape(-1) / 255.0 for img in img_list]
    ).astype(np.float32)
    return img_inputs


class L2SimMetric(BaseSimMetric):
    def __init__(
        self,
        is_calibrated=True,
        s_nn=1,
        dist_threshold=0.8,
        is_ascend=True,
        index_type="euclidean",
        batch_size=50,
    ):
        # init with hypers
        super().__init__(
            dist_threshold,
            is_ascend,
            index_type,
            batch_size,
        )

        self.is_calibrated = is_calibrated
        if self.is_calibrated:
            self.s_nn = s_nn

    def feature_transform(self, reference_list):
        """
        transform list of img path to the features that are used for memorization check
        """
        reference_imgs = get_img_pixel_input(reference_list)
        return reference_imgs

    def calibrated_l2_distance(self, query):
        """
        compute calibrated l2 distance in batch
        Parameters:
            self:
                self.index: faiss index for fast nearest neighbour search
                self.reference_imgs: the list of reference imgs, shape (refset_size, input_dim)
                self.s_nn: size of the nearest neighbour set
            query: a list of query imgs, shape (batch_size, input_dim)
        Return:
            distance_list: a list of distance among query and the closest reference sample
            closest_ref_list: a list of refimgs (path) that matched
        """
        # search for closest reference samples first
        Dist, Idx = self.index.search(query, 1)  # D shape (query_bs, s_nn)
        Dist = np.sqrt(Dist)

        # based on the extract reference samples, finding the nearest neighbours
        closest_idx_list = Idx[:, 0]
        extract_refimg_list = np.array(
            [self.reference_imgs[closest_idx] for closest_idx in closest_idx_list]
        )
        closest_dis_list = Dist[:, 0]
        D_nn, I_nn = self.index.search(extract_refimg_list, self.s_nn + 1)

        # exclude itself
        D_nn = D_nn[:, 1:]
        D_nn = np.sqrt(D_nn)

        # compute the calibrated l2 distance; normalized distance
        distance_list = closest_dis_list / D_nn.mean(axis=-1)
        closest_ref_list = [self.reference_list[closest_idx] for closest_idx in closest_idx_list]
        return distance_list, closest_ref_list

    def absolute_l2_distance(self, query):
        """
        compute absolute l2 distance in batch
        Parameters:
            self:
                self.index: faiss index for fast nearest neighbour search
                self.trainset_img: the list of reference imgs, shape (refset_size, input_dim)
            query: a list of query imgs, shape (batch_size, input_dim)
        """
        # search for closest reference samples first
        Dist, Idx = self.index.search(query, 1)  # D shape (query_bs, 1)
        Dist = np.sqrt(Dist)
        closest_idx_list = Idx[:, 0]
        closest_dis_list = Dist[:, 0]
        closest_ref_list = [self.reference_list[closest_idx] for closest_idx in closest_idx_list]

        return closest_dis_list, closest_ref_list

    def compute_pairwise_distance(self, query):
        if self.is_calibrated:
            # calibrated l2 distance
            distance_list, closest_ref_list = self.calibrated_l2_distance(query)
        else:
            # absolute l2 distance
            distance_list, closest_ref_list = self.absolute_l2_distance(query)
        return distance_list, closest_ref_list
