import os
import pathlib

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {"bmp", "jpg", "jpeg", "pgm", "png", "ppm", "tif", "tiff", "webp"}


def get_images_in_dir(path):
    """Return list of all files at path of type IMAGE_EXTENSIONS"""
    files = sorted([file for ext in IMAGE_EXTENSIONS for file in path.rglob(f"*.{ext}")])

    return files


def get_image_dataset_in_dir(path):
    """
    Get all the images under the given local filesystem path
    Params:
        path: path to the saved images
    Return:
        files: a list of filenames (path)
        labels: a list of class labels, empty if no sub-folders
    """
    image_path = pathlib.Path(path)
    files = get_images_in_dir(image_path)
    labels = []
    class_idx = 0

    def get_order(file):
        filename = os.path.splitext(os.path.basename(file))[0]
        return int(filename)

    if not files:
        # Assume sub-folders for image classes
        class_dirs = sorted(
            image_path.glob("*"), key=get_order
        )  # look for all subfolders in the numerical order
        files = []
        for f in class_dirs:
            files_in_path = get_images_in_dir(f)
            files += files_in_path
            labels.extend([class_idx for _ in range(len(files_in_path))])
            class_idx += 1
    labels = np.array(labels, dtype=np.int32)

    return files, labels


def concat_imgpair(query_path, refimg_path):
    """
    Concatenate query img and its closest reference image
    """
    final_img = np.array(Image.open(query_path))
    ref_img = np.array(Image.open(refimg_path))
    final_img = np.concatenate((final_img, ref_img), axis=0)
    return final_img


def save_matched_images(matched_results, save_path):
    # concatenate and save results
    query_cnt = 0
    collection_img = None
    row_img = None
    for query_path, (distance, refimg_path) in matched_results:
        paired_img = concat_imgpair(query_path, refimg_path)
        query_cnt += 1
        if row_img is None:
            row_img = paired_img
        else:
            row_img = np.concatenate((row_img, paired_img), axis=1)
        if query_cnt % 30 == 0:
            if collection_img is None:
                collection_img = row_img
            else:
                collection_img = np.concatenate((collection_img, row_img), axis=0)
            row_img = None

    if collection_img is None and row_img is not None:
        collection_img = row_img
        row_img = None

    if collection_img is None:
        print("No img is matched ! ")
    else:
        im = Image.fromarray(np.uint8(collection_img))
        im.save(save_path)


class ImagesFromFilenames(Dataset):
    def __init__(self, image_files, transform=None):
        self.image_files = image_files
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image
