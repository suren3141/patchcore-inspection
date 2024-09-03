import csv
import logging
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import tqdm

LOGGER = logging.getLogger(__name__)


def plot_segmentation_images(
    savefolder,
    image_paths,
    segmentations,
    anomaly_scores=None,
    anomaly_labels=None,
    mask_paths=None,
    image_transform=lambda x: x,
    mask_transform=lambda x: x,
    save_depth=4,
):
    """Generate anomaly segmentation images.

    Args:
        image_paths: List[str] List of paths to images.
        segmentations: [List[np.ndarray]] Generated anomaly segmentations.
        anomaly_scores: [List[float]] Anomaly scores for each image.
        mask_paths: [List[str]] List of paths to ground truth masks.
        image_transform: [function or lambda] Optional transformation of images.
        mask_transform: [function or lambda] Optional transformation of masks.
        save_depth: [int] Number of path-strings to use for image savenames.
    """
    if mask_paths is None or not any(mask_paths):
        mask_paths = ["-1" for _ in range(len(image_paths))]
    masks_provided = mask_paths[0] != "-1"
    if anomaly_scores is None or not any(anomaly_scores):
        anomaly_scores = ["-1" for _ in range(len(image_paths))]

    os.makedirs(os.path.join(savefolder, "output_images"), exist_ok=True)

    for image_path, mask_path, anomaly_score, anomaly_label, segmentation in tqdm.tqdm(
        zip(image_paths, mask_paths, anomaly_scores, anomaly_labels, segmentations),
        total=len(image_paths),
        desc="Generating Segmentation Images...",
        leave=False,
    ):
        image = PIL.Image.open(image_path).convert("RGB")
        image = image_transform(image)
        if not isinstance(image, np.ndarray):
            image = image.numpy()

        if masks_provided and mask_path not in [None, "-1"]:
            mask = PIL.Image.open(mask_path).convert("RGB")
            mask = mask_transform(mask)
            if not isinstance(mask, np.ndarray):
                mask = mask.numpy()
        else:
            mask = np.zeros_like(image)

        savename = image_path.split("/")
        savename = "_".join(savename[-save_depth:])
        savename = os.path.join(savefolder, "output_images", savename)
        f, axes = plt.subplots(1, 3)
        axes[0].set_title(f"anomaly:{anomaly_label}")
        axes[0].imshow(image.transpose(1, 2, 0))
        axes[1].imshow(mask.transpose(1, 2, 0))
        axes[1].set_title(f"mask:{masks_provided}")
        axes[2].imshow(segmentation)
        axes[2].set_title(f"{anomaly_score:.3f}")
        f.set_size_inches(3 * 3, 3)
        f.tight_layout()
        f.savefig(savename)
        plt.close()

import json

def save_anomaly_scores(
    savefolder,
    image_paths,
    anomaly_scores=None,
):
    """Save anomaly scares.

    Args:
        image_paths: List[str] List of paths to images.
        anomaly_scores: [List[float]] Anomaly scores for each image.
    """
    if anomaly_scores is None or not any(anomaly_scores):
        raise NotImplementedError

    os.makedirs(savefolder, exist_ok=True)

    score_dict = {
        image_path : float(anomaly_score) for image_path, anomaly_score in zip(image_paths, anomaly_scores)
    }

    with open(os.path.join(savefolder, "scores.json"), "w+") as f:
        json.dump(score_dict, f, indent=4)


def create_storage_folder(
    main_folder_path, project_folder, group_folder, mode="iterate"
):
    os.makedirs(main_folder_path, exist_ok=True)
    project_path = os.path.join(main_folder_path, project_folder)
    os.makedirs(project_path, exist_ok=True)
    save_path = os.path.join(project_path, group_folder)
    if mode == "iterate":
        counter = 0
        while os.path.exists(save_path):
            save_path = os.path.join(project_path, group_folder + "_" + str(counter))
            counter += 1
        os.makedirs(save_path)
    elif mode == "overwrite":
        os.makedirs(save_path, exist_ok=True)

    return save_path


def set_torch_device(gpu_ids):
    """Returns correct torch.device.

    Args:
        gpu_ids: [list] list of gpu ids. If empty, cpu is used.
    """
    if len(gpu_ids):
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        return torch.device("cuda:{}".format(gpu_ids[0]))
    return torch.device("cpu")


def fix_seeds(seed, with_torch=True, with_cuda=True):
    """Fixed available seeds for reproducibility.

    Args:
        seed: [int] Seed value.
        with_torch: Flag. If true, torch-related seeds are fixed.
        with_cuda: Flag. If true, torch+cuda-related seeds are fixed
    """
    random.seed(seed)
    np.random.seed(seed)
    if with_torch:
        torch.manual_seed(seed)
    if with_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def compute_and_store_final_results(
    results_path,
    results,
    row_names=None,
    column_names=[
        "Instance AUROC",
        "Full Pixel AUROC",
        "Full PRO",
        "Anomaly Pixel AUROC",
        "Anomaly PRO",
    ],
):
    """Store computed results as CSV file.

    Args:
        results_path: [str] Where to store result csv.
        results: [List[List]] List of lists containing results per dataset,
                 with results[i][0] == 'dataset_name' and results[i][1:6] =
                 [instance_auroc, full_pixelwisew_auroc, full_pro,
                 anomaly-only_pw_auroc, anomaly-only_pro]
    """
    if row_names is not None:
        assert len(row_names) == len(results), "#Rownames != #Result-rows."

    mean_metrics = {}
    for i, result_key in enumerate(column_names):
        mean_metrics[result_key] = np.mean([x[i] for x in results])
        LOGGER.info("{0}: {1:3.3f}".format(result_key, mean_metrics[result_key]))

    def get_savename(results_path, mode="iterate"):

        savename = os.path.join(results_path, "results.csv")

        if mode == "iterate":
            counter = 0
            while os.path.exists(savename):
                savename = os.path.join(results_path, f"results_{counter}.csv")
                counter += 1
        elif mode == "overwrite":
            pass

        return savename

    savename = get_savename(results_path)
    # savename = os.path.join(results_path, "results.csv")

    print(f"saving results : {savename}")

    with open(savename, "w") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",")
        header = column_names
        if row_names is not None:
            header = ["Row Names"] + header

        csv_writer.writerow(header)
        for i, result_list in enumerate(results):
            csv_row = result_list
            if row_names is not None:
                csv_row = [row_names[i]] + result_list
            csv_writer.writerow(csv_row)
        mean_scores = list(mean_metrics.values())
        if row_names is not None:
            mean_scores = ["Mean"] + mean_scores
        csv_writer.writerow(mean_scores)

    mean_metrics = {"mean_{0}".format(key): item for key, item in mean_metrics.items()}
    return mean_metrics
