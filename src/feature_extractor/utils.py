from pathlib import Path
file_path =  Path(__file__).absolute()
src_dir = file_path.parents[1]

import sys
sys.path.append(src_dir)

from patchcore.common import NetworkFeatureAggregator
from patchcore.datasets.monuseg import MoNuSegDataset

from tqdm import tqdm
from torch.utils.data import DataLoader
import tracemalloc

from collections import defaultdict
import numpy as np
import torch
from typing import List

def get_optimus_backbone(seed):
    from feature_extractor.optimus import load_optimus

    backbone = load_optimus()
    backbone.name, backbone.seed = "optimus", seed

    return backbone

def get_medsam_backbone(seed, model_weights_path = "/mnt/dataset/medsam/medsam_vit_b.pth"):
    from segment_anything import sam_model_registry
    from types import MethodType

    sam = sam_model_registry["vit_b"](checkpoint=model_weights_path)
    # predictor = SamPredictor(sam)
    backbone = sam
    backbone.name, backbone.seed = "medsam", seed

    def _preprocess(self, x):
        # Removed normalize from SAM preprocess since data is already normalized
        from torch.nn import functional as F
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x            
    
    backbone.preprocess = MethodType(_preprocess, backbone)

    return backbone


def get_backbone(backbone_name, backbone_seed=None):

    if backbone_name == "optimus":
        backbone = get_optimus_backbone(backbone_seed)
    elif backbone_name == "optimus_old":
        from feature_extractor.optimus import load_optimus_old

        backbone = load_optimus_old()
        backbone.name, backbone.seed = "optimus", backbone_seed
    elif backbone_name == "medsam":
        backbone = get_medsam_backbone(backbone_seed)

    elif backbone_name == "inception_v3":
        from pytorch_fid.inception import InceptionV3
        backbone = InceptionV3()
        backbone.name, backbone.seed = backbone_name, backbone_seed

    elif backbone_name == "ResNet50":
        from torchvision.models import resnet50, ResNet50_Weights
        weights = ResNet50_Weights.DEFAULT
        backbone = resnet50(weights=weights)
        backbone.name, backbone.seed = backbone_name, backbone_seed
        # backbone.transforms = weights.transforms()
        # print(backbone.transforms)

    elif backbone_name == "ResNet18":
        from torchvision.models import resnet18, ResNet18_Weights
        weights = ResNet18_Weights.DEFAULT
        backbone = resnet18(weights=weights)
        backbone.name, backbone.seed = backbone_name, backbone_seed
        backbone.transforms = weights.transforms()
        print(backbone.transforms)

    else:
        import sys
        sys.path.append('src/patchcore')
        import patchcore.backbones

        backbone = patchcore.backbones.load(backbone_name)
        backbone.name, backbone.seed = backbone_name, backbone_seed

    return backbone

def compute_features(dataloader, feature_aggregator, layers, device, flat=True):
    # Start tracing memory allocation
    tracemalloc.start()

    features_dict = defaultdict(list)

    with tqdm(total=len(dataloader), desc='Computing features') as pbar:
        for image in dataloader:
            if isinstance(image, dict):
                image = image["image"]

            with torch.no_grad():
                input_image = image.to(torch.float).to(device)

                feat = feature_aggregator(input_image)
                for layer in layers:
                    feat_layer = feat[layer].detach().cpu().numpy()
                    # TODO : Do I need to squeeze here?
                    # feat_layer = feat_layer.squeeze()
                    if flat and feat_layer.ndim != 2:
                        feat_layer = np.mean(feat_layer, axis=(-1, -2))

                    features_dict[layer].append(feat_layer)

                current, peak = tracemalloc.get_traced_memory()

                # Update the custom tqdm bar to display memory usage
                pbar.set_description(f'Memory Usage: {current / 10**6:.2f} MB')
                pbar.update(1)

    for layer in layers:
        features_dict[layer] = np.concatenate(features_dict[layer], axis=0)


def extract_features(
    images : List,
    backbone_name : str,
    device,
    layers : List = ["out"], 
    batch_size : int = 2, 
    num_workers : int = 1, 
    norm : bool = True, 
    flat : bool =True
    ):

    # assert backbone_name in ['optimus', 'medsam', 'inception_v3']
    # assert 'out' in layers

    # norm = backbone_name in ["medsam"]

    if backbone_name in ["optimus", "optimus_old"]:
        resize = 224
    else:
        resize = 0

    print("Loading data...")
    # TODO : image size and resize set to 0 to avoid scaling and cropping 
    image_dataset = MoNuSegDataset(images, cropsize=0, resize=resize, norm=norm)
    dataloader = DataLoader(image_dataset, batch_size=batch_size, shuffle=False)

    print("Loading backbone...")
    backbone = get_backbone(backbone_name)
    backbone = backbone.to(device)

    feature_aggregator = NetworkFeatureAggregator(backbone, layers, device)

    print("Computing features...")
    features_dict = compute_features(dataloader, feature_aggregator, layers, device, flat=flat)



    return features_dict
