import timm
import functools
import time
from collections import OrderedDict


# Install from https://github.com/facebookresearch/segment-anything
# from segment_anything import SamPredictor, sam_model_registry
from tqdm.notebook import tqdm
import torch
import torch.nn as nn

import numpy as np
from pathlib import Path

import functools
from torchvision import transforms 
from torch.utils.data import Dataset
from torchvision.io import read_image

from torch.utils.data import DataLoader
import glob, os

from types import MethodType


def load_optimus(PATH_TO_CHECKPOINT = "/mnt/dataset/h_optimus_0/checkpoint.pth", device='cuda'):

    params = {
        "patch_size": 14,
        "embed_dim": 1536,
        "depth": 40,
        "num_heads": 24,
        "init_values": 1e-05,
        "mlp_ratio": 5.33334,
        "mlp_layer": functools.partial(
            timm.layers.mlp.GluMlp,
            act_layer=torch.nn.modules.activation.SiLU,
            gate_last=False,
        ),
        "act_layer": torch.nn.modules.activation.SiLU,
        "reg_tokens": 4,
        "no_embed_class": True,
        "img_size": 224,
        "num_classes": 0,
        "in_chans": 3,
    }

    if os.path.exists(PATH_TO_CHECKPOINT):

        model_h_optimus = timm.models.VisionTransformer(**params)

        # load state dict from checkpoint
        checkpoint = torch.load(PATH_TO_CHECKPOINT, map_location="cpu")
        # checkpoint = torch.load(PATH_TO_CHECKPOINT, weights_only=True)

        model_h_optimus.load_state_dict(checkpoint, strict=True)
    else:
        model_h_optimus = timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True, init_values=1e-5, dynamic_img_size=False)


    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed.proj(x)
        return x

    model_h_optimus._process_input = MethodType(_process_input, model_h_optimus)

    model_h_optimus.eval()
    model_h_optimus.to(device)

    return model_h_optimus




def load_optimus_old(PATH_TO_CHECKPOINT = "/mnt/dataset/h_optimus_0/checkpoint.pth", device='cuda'):
    params = {
        'patch_size': 14, 
        'embed_dim': 1536, 
        'depth': 40, 
        'num_heads': 24, 
        'init_values': 1e-05, 
        'mlp_ratio': 5.33334, 
        'mlp_layer': functools.partial(
            timm.layers.mlp.GluMlp, act_layer=torch.nn.modules.activation.SiLU, gate_last=False
        ), 
        'act_layer': torch.nn.modules.activation.SiLU, 
        'reg_tokens': 4, 
        # 'reg_tokens': 0, 
        # 'class_token': False,
        # 'global_pool' : '',
        'no_embed_class': True, 
        # 'img_size': 128, 
        'img_size': 224, 
        'num_classes': 0, 
        'in_chans': 3
    }

    if os.path.exists(PATH_TO_CHECKPOINT):

        model_h_optimus = timm.models.VisionTransformer(**params)

        # load state dict from checkpoint
        checkpoint = torch.load(PATH_TO_CHECKPOINT, map_location="cpu")
        # checkpoint = torch.load(PATH_TO_CHECKPOINT, weights_only=True)

        model_h_optimus.load_state_dict(checkpoint, strict=True)
    else:
        model_h_optimus = timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True, init_values=1e-5, dynamic_img_size=False)

    model_h_optimus.eval()
    model_h_optimus.to(device)

    return model_h_optimus


def get_h_optimus_feat(dataloader, PATH_TO_CHECKPOINT = "/mnt/dataset/h_optimus_0/checkpoint.pth", model=None, batch_size=1):

    print("Loading model...")

    if model is None:

        model = load_optimus_old()

    model.load_state_dict(torch.load(PATH_TO_CHECKPOINT, map_location="cpu"))

    model.eval()
    model.to("cuda")

    feats = []


    print("Extracting features...")

    # We recommend using mixed precision for faster inference.
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        with torch.inference_mode():
            for data in tqdm(dataloader):
                # print(data.shape)
                features = model(data.to("cuda"))
                # print(features.shape)


                assert features.shape == (data.shape[0], 1536)

                feats.append(features.detach().cpu().numpy())


        # feats.append(feat)

    print(len(feats))

    return np.concatenate(feats, axis=0)





# if __name__ == "__main__":
#     params = {
#         'patch_size': 14, 
#         'embed_dim': 1536, 
#         'depth': 40, 
#         'num_heads': 24, 
#         'init_values': 1e-05, 
#         'mlp_ratio': 5.33334, 
#         'mlp_layer': functools.partial(
#             timm.layers.mlp.GluMlp, act_layer=torch.nn.modules.activation.SiLU, gate_last=False
#         ), 
#         'act_layer': torch.nn.modules.activation.SiLU, 
#         'reg_tokens': 4, 
#         'no_embed_class': True, 
#         # 'img_size': 128, 
#         'img_size': 224, 
#         'num_classes': 0, 
#         'in_chans': 3
#     }

#     model_h_optimus = timm.models.VisionTransformer(**params)
