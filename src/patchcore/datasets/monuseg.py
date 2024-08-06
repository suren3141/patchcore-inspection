import os, glob
from enum import Enum

import PIL
import torch
from torchvision import transforms

from pathlib import Path


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class MoNuSegDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for MoNuSeg.
    """

    # TODO : change this
    transform_mean=(0.707223, 0.578729, 0.703617)
    transform_std=(0.211883, 0.230117, 0.177517)

    def __init__(
        self,
        data_path,
        resize=256,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        train_val_split=1.0,
        **kwargs,
    ):
        """
        Args:
            data_path: [str]. Path to the MoNuSeg data folder.
            resize: [int]. (Square) Size the loaded image initially gets resized to.
            imagesize: [int]. (Square) Size the resized loaded image gets (center-)cropped to.
            split: [enum-option]. Indicates if training or test split of the
                   data should be used. Has to be an option taken from
                   DatasetSplit, e.g. mvtec.DatasetSplit.TRAIN. Note that
                   mvtec.DatasetSplit.TEST will also load mask data.
        """
        super().__init__()
        self.data_path = data_path
        self.split = split
        self.train_val_split = train_val_split

        self.img_files = self._get_img_files()

        self.data_to_iterate = self._get_data_to_iterate()

        # self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        self.transform_img = [
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=MoNuSegDataset.transform_mean, std=MoNuSegDataset.transform_std),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        # self.transform_mask = [
        #     transforms.Resize(resize),
        #     transforms.CenterCrop(imagesize),
        #     transforms.ToTensor(),
        # ]
        # self.transform_mask = transforms.Compose(self.transform_mask)
        self.transform_mask = None

        self.imagesize = (3, imagesize, imagesize)

    def _get_img_files(self):

        par_path = os.path.join(self.data_path, self.split.value)
        data_types = os.listdir(par_path)

        for data_type in data_types:
            path = os.path.join(par_path, data_type)
            img_files = glob.glob(os.path.join(path, "*.png"))

            self.img_files = img_files


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

    def __len__(self):
        return len(self.data_to_iterate)

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
            # TODO : Change this
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
