import sys
sys.path.append('./src')

from patchcore.common import NetworkFeatureAggregator
from patchcore.datasets.monuseg import get_monuseg_images, MoNuSegDataset

from tqdm.notebook import tqdm
from torch.utils.data import DataLoader

import numpy as np
import torch

def get_backbone(backbone_name, backbone_seed=None):

    if backbone_name == "optimus":
        from patchcore.optimus import load_optimus

        backbone = load_optimus()
        backbone.name, backbone.seed = backbone_name, backbone_seed
    elif backbone_name == "medsam":
        from segment_anything import SamPredictor, sam_model_registry
        from types import MethodType

        model_weights_path = "/mnt/dataset/medsam/medsam_vit_b.pth"
        sam = sam_model_registry["vit_b"](checkpoint=model_weights_path)
        # predictor = SamPredictor(sam)
        backbone = sam
        backbone.name, backbone.seed = backbone_name, backbone_seed

        def _preprocess(self, x):
            # Removed normalize from SAM preprocess since data is already normalized
            from torch.nn import functional as F
            h, w = x.shape[-2:]
            padh = self.image_encoder.img_size - h
            padw = self.image_encoder.img_size - w
            x = F.pad(x, (0, padw, 0, padh))
            return x            
        
        backbone.preprocess = MethodType(_preprocess, backbone)

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

def extract_features(images, backbone_name, device, layers=["out"], batch_size=2, num_workers=1, norm=True):

    # assert backbone_name in ['optimus', 'medsam', 'inception_v3']
    # assert 'out' in layers

    # norm = backbone_name in ["medsam"]

    if backbone_name == "optimus":
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

    features = []
    for image in tqdm(dataloader, desc='computing features'):
        if isinstance(image, dict):
            image = image["image"]
        with torch.no_grad():
            input_image = image.to(torch.float).to(device)

            feat = feature_aggregator(input_image)
            feat = [feat[layer] for layer in layers]
            feat = [x.detach().cpu().numpy() for x in feat]
            # print(len(feat), feat[0].shape)

            features.append(feat)

    out = {}
    for ind, l in enumerate(layers):
        out[l] = np.concatenate([f[ind] for f in features], axis=0)

    return out
