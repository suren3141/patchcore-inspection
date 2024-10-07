import os, glob
from enum import Enum

import PIL
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from pathlib import Path



class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class MoNuSegDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for MoNuSeg.
    """

    # TODO : change this
    # transform_mean=(0.707223, 0.578729, 0.703617)
    # transform_std=(0.211883, 0.230117, 0.177517)

    # Updated mean, std for monuseg
    # transform_mean=[0.6910, 0.4947, 0.6422]
    # transform_std=[0.1662, 0.1828, 0.1421]

    transform_mean= [0.6444, 0.4477, 0.6041]
    transform_std = [0.1820, 0.1836, 0.1473]


    def __init__(
        self,
        data_path,
        norm=True,
        resize=256,
        cropsize=224,
        split=DatasetSplit.TRAIN,
        train_val_split=1.0,
        **kwargs,
    ):
        """
        Args:
            data_path: [str]. Path to the MoNuSeg data folder.
            resize: [int]. (Square) Size the loaded image initially gets resized to.
            cropsize: [int]. (Square) Size the resized loaded image gets (center-)cropped to.
            split: [enum-option]. Indicates if training or test split of the
                   data should be used. Has to be an option taken from
                   DatasetSplit, e.g. mvtec.DatasetSplit.TRAIN. Note that
                   mvtec.DatasetSplit.TEST will also load mask data.
        """
        super().__init__()
        self.split = split
        self.train_val_split = train_val_split

        if isinstance(data_path, str):
            self.data_path = data_path
            self.img_files = self._get_img_files()

        elif isinstance(data_path, list):
            self.data_path = None
            self.img_files = data_path

        else:
            raise NotImplementedError

        
        self.data_to_iterate = self._get_data_to_iterate()

        transform = []
        if resize != 0:
            transform.append(transforms.Resize(resize))
            self.imagesize = (3, resize, resize)
        if cropsize != 0 and cropsize != resize:
            transform.append(transforms.CenterCrop(cropsize))
            self.imagesize = (3, cropsize, cropsize)


        transform += [
            # transforms.ToPILImage(),
            transforms.ToTensor(),
        ]

        if norm:
            transform.append(
                transforms.Normalize(
                    mean=MoNuSegDataset.transform_mean, 
                    std=MoNuSegDataset.transform_std
                )
            )

        self.transform_img = transforms.Compose(transform)


    def _get_img_files(self):

        raise NotImplementedError()

        par_path = os.path.join(self.data_path, self.split.value)
        data_types = os.listdir(par_path)

        for data_type in data_types:
            path = os.path.join(par_path, data_type)
            img_files = glob.glob(os.path.join(path, "*.png"))

            self.img_files = img_files

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        # image = read_image(img_path)
        image = self._load_img(img_path)
        if self.transform_img:
            image = self.transform_img(image)
        return image
    
    '''

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask = PIL.Image.open(mask_path)
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        return {
            "image": image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            "image_name": "/".join(image_path.split("/")[-4:]),
            "image_path": image_path,
        }
        '''

    def _load_img(self, path):
        from PIL import Image
        import numpy as np
        import blobfile as bf

        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        return pil_image

    def _get_data_to_iterate(self):

        data_to_iterate = []
        for img_name in self.img_files:
            path = Path(img_name)
            classname = path.parent.absolute().stem
            anomaly= "good" if classname in ["good", "gt"] else "syn"
            image_path = img_name
            mask_path = None            
            data_tuple = [classname, anomaly, image_path, mask_path]
            data_to_iterate.append(data_tuple)

        return data_to_iterate

    def get_image_data(self):


        imgpaths_per_class = {}
        maskpaths_per_class = {}

        for classname in self.classnames_to_use:
            classpath = os.path.join(self.data_path, classname, self.split.value)
            maskpath = os.path.join(self.data_path, classname, "ground_truth")
            anomaly_types = os.listdir(classpath)

            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}

            for anomaly in anomaly_types:
                anomaly_path = os.path.join(classpath, anomaly)
                anomaly_files = sorted(os.listdir(anomaly_path))
                imgpaths_per_class[classname][anomaly] = [
                    os.path.join(anomaly_path, x) for x in anomaly_files
                ]

                if self.train_val_split < 1.0:
                    n_images = len(imgpaths_per_class[classname][anomaly])
                    train_val_split_idx = int(n_images * self.train_val_split)
                    if self.split == DatasetSplit.TRAIN:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                            classname
                        ][anomaly][:train_val_split_idx]
                    elif self.split == DatasetSplit.VAL:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                            classname
                        ][anomaly][train_val_split_idx:]

                if self.split == DatasetSplit.TEST and anomaly != "good":
                    anomaly_mask_path = os.path.join(maskpath, anomaly)
                    anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
                    maskpaths_per_class[classname][anomaly] = [
                        os.path.join(anomaly_mask_path, x) for x in anomaly_mask_files
                    ]
                else:
                    maskpaths_per_class[classname]["good"] = None

        # Unrolls the data dictionary to an easy-to-iterate list.
        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    data_tuple = [classname, anomaly, image_path]
                    if self.split == DatasetSplit.TEST and anomaly != "good":
                        data_tuple.append(maskpaths_per_class[classname][anomaly][i])
                    else:
                        data_tuple.append(None)
                    data_to_iterate.append(data_tuple)

        return imgpaths_per_class, data_to_iterate

import numpy as np
import time
from typing import List

def get_monuseg_images_recursive(data_path : str):
    files = []
    for entry in os.listdir(data_path):
        full_path = os.path.join(data_path, entry)
        if os.path.isdir(full_path):
            files += get_monuseg_images_recursive(full_path)
        elif full_path.endswith('.png'):
            files.append(full_path)

    return files

def get_monuseg_images(data_path : str, directories : List, subsample=None, verbose=False):

    images = []

    for d in directories:
        images_ = glob.glob(os.path.join(data_path, d, "*.png"))

        if subsample:
            t = 1000 * time.time() # current time in milliseconds
            np.random.seed(int(t) % 2**32)

            images_ = list(np.random.choice(images_, int(len(images_)*subsample), replace=False))


        if verbose: print(d, '->', hash(''.join(images_)))

        images+= images_

    if verbose: print('->', hash(''.join(images)))

    return images



def get_monuseg_dataloader(data_path, batch_size=1, split="", cropsize=224, resize=256, subsample=None):

    if split == "test":

        # data_dirs = ["train/gt", "test/gt", "test/syn/v1.2_*/samples/", "test/syn/v1.3_*/samples/", "test/syn/v1.4_*/samples/"]
        # images = get_monuseg_images(data_path, data_dirs, subsample=subsample)

        data_path = os.path.join(data_path, split, "gt")
        images_gt = get_monuseg_images_recursive(data_path)

        data_path = os.path.join(data_path, split, "syn")
        images_syn = get_monuseg_images_recursive(data_path)

        images = images_gt + images_syn

        if subsample:
            t = 1000 * time.time() # current time in milliseconds
            np.random.seed(int(t) % 2**32)

            images = list(np.random.choice(images, int(len(images)*subsample), replace=False))

        
    else:
        data_path = os.path.join(data_path, split, "gt")
        # images = glob.glob(os.path.join(data_path, split, "gt", "*.png"))
        images = get_monuseg_images_recursive(data_path)

    # TODO : Update this to take in only path (and not images)
    image_dataset = MoNuSegDataset(images, cropsize=cropsize, resize=resize)
    dataloader = DataLoader(image_dataset, batch_size=batch_size, shuffle=False)

    return dataloader
