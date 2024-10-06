import sys, os
sys.path.append('./src')

import patchcore.utils
import numpy as np

def _save_segmentation_images(
        run_save_path,
        dataloader,
        segmentations,
        scores,
        ):

    dataset_name = dataloader.dataset.name

    anomaly_labels = [
        x[1] != "good" for x in dataloader.dataset.data_to_iterate
    ]

    image_paths = [
        x[2] for x in dataloader.dataset.data_to_iterate
    ]
    mask_paths = [
        x[3] for x in dataloader.dataset.data_to_iterate
    ]

    def image_transform(image):
        in_std = np.array(
            dataloader.dataset.transform_std
        ).reshape(-1, 1, 1)
        in_mean = np.array(
            dataloader.dataset.transform_mean
        ).reshape(-1, 1, 1)
        image = dataloader.dataset.transform_img(image)
        return np.clip(
            (image.numpy() * in_std + in_mean) * 255, 0, 255
        ).astype(np.uint8)

    def mask_transform(mask):
        return dataloader.dataset.transform_mask(mask).numpy()

    image_save_path = os.path.join(
        run_save_path, "segmentation_images", dataset_name
    )
    os.makedirs(image_save_path, exist_ok=True)
    patchcore.utils.plot_segmentation_images(
        image_save_path,
        image_paths,
        segmentations,
        scores,
        anomaly_labels,
        mask_paths,
        image_transform=image_transform,
        mask_transform=mask_transform,
    )
