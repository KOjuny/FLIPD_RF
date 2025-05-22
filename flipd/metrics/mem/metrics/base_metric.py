from abc import ABC, abstractmethod

import faiss
from tqdm import tqdm

from .utils import save_matched_images


class BaseSimMetric(ABC):
    def __init__(
        self,
        dist_threshold,
        is_ascend,
        index_type,
        batch_size,
    ):
        self.dist_threshold = dist_threshold
        self.is_ascend = is_ascend
        self.index_type = index_type
        self.batch_size = batch_size

    @abstractmethod
    def feature_transform(
        self,
    ):
        """
        transform list of img path to the features that are used for memorization check
        """

    @abstractmethod
    def compute_pairwise_distance(self, query):
        """
        given query, compute the distance with the trainset and return the dist_list and imgpath_list of the closest training imgs
        """

    def build_reference_database(self, reference_list):
        """build the index on training set using faiss to accelerate searching"""

        self.reference_list = reference_list

        # load the training images (pixel flatten)
        self.reference_imgs = self.feature_transform(reference_list)

        # build faiss index
        d = self.reference_imgs.shape[-1]
        if self.index_type == "euclidean":
            self.index = faiss.IndexFlatL2(d)
        elif self.index_type == "inner_product":
            self.index = faiss.IndexFlatIP(d)
        else:
            raise NotImplementedError(f"Index type {self.index} not implemented !")

        self.index.add(self.reference_imgs)

    def get_ranked_distances(self, queryset_list, distance_save_path=None):
        """
        Iterate over the batches of query samples
        """
        query_nn_dict = {}
        query_num = len(queryset_list)
        for qidx in tqdm(range(0, query_num, self.batch_size)):
            # Get query features (bs, feat_dim)
            query = self.feature_transform(queryset_list[qidx : qidx + self.batch_size])

            # Compute distance and get the closest sample
            distance_list, closest_ref_list = self.compute_pairwise_distance(query)

            for distance, closest_ref_img, query_imgp in zip(
                distance_list,
                closest_ref_list,
                queryset_list[qidx : qidx + self.batch_size],
            ):
                # dict[key]: (Distance, refimg_path)
                query_nn_dict[query_imgp] = (distance, closest_ref_img)

        # sort the matched samples by distance; List of (query_img, (distance, ref_img))
        if self.is_ascend:
            sorted_results = sorted(query_nn_dict.items(), key=lambda item: item[1][0])
        else:
            sorted_results = sorted(query_nn_dict.items(), key=lambda item: -item[1][0])

        # save intermediate results
        if distance_save_path is not None:
            import pickle as pkl

            with open(distance_save_path, "wb") as handle:
                pkl.dump(query_nn_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)

        return sorted_results

    def filter_memorization_with_threshold(self, sorted_results):
        """
        filtered the matched samples based on the threshold
        Parameters:
            sorted_results: list of (query_path, (distance, ref_path)) tuple
            is_ascend: if True, then the distance rank is starting with smallest value; if False, starting with the largest value
        """
        #
        tot_cnt = len(sorted_results)
        matched_cnt = 0
        while matched_cnt < tot_cnt:
            query_path, (distance, ref_path) = sorted_results[matched_cnt]
            if self.is_ascend:
                if distance < self.dist_threshold:
                    matched_cnt += 1
                else:
                    break
            else:
                if distance > self.dist_threshold:
                    matched_cnt += 1
                else:
                    break

        print(f"Total number: {tot_cnt} of queries ")
        print(f"Matched number: {matched_cnt} queries matched")

        return sorted_results[:matched_cnt]

    def visualize(self, matched_results, vis_num=None, save_path="matched_images.png"):
        # concatenate results
        if vis_num is None:
            vis_num = len(matched_results)

        save_matched_images(matched_results[:vis_num], save_path)
