import timm
import functools
import time


# Install from https://github.com/facebookresearch/segment-anything
# from segment_anything import SamPredictor, sam_model_registry
from tqdm.notebook import tqdm
import torch
import numpy as np
from pathlib import Path

import functools

import timm
import torch
from torchvision import transforms 
from torch.utils.data import Dataset
from torchvision.io import read_image

'''
class ImageDataset(Dataset):

    transform_mean=(0.707223, 0.578729, 0.703617)
    transform_std=(0.211883, 0.230117, 0.177517)

    def __init__(self, img_files, imagesize=None, resize=None):

        self.img_files = img_files

        transform = []
        if resize != 0:
            transform.append(transforms.Resize(resize))
            self.imagesize = (3, resize, resize)
        if imagesize != 0:
            transform.append(transforms.CenterCrop(imagesize))
            self.imagesize = (3, imagesize, imagesize)


        transform += [
            # transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=ImageDataset.transform_mean, 
                std=ImageDataset.transform_std
            ),
        ]

        self.transform_img = transforms.Compose(transform)


        self.data_to_iterate = self._get_data_to_iterate()

    def _get_data_to_iterate(self):

        data_to_iterate = []
        for img_name in self.img_files:
            path = Path(img_name)
            classname = path.parent.absolute().stem
            anomaly= "good" if classname in ["train", "gt"] else "syn"
            image_path = img_name
            mask_path = None            
            data_tuple = [classname, anomaly, image_path, mask_path]
            data_to_iterate.append(data_tuple)

        return data_to_iterate


    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        # image = read_image(img_path)
        image = self.load_img(img_path)
        if self.transform_img:
            image = self.transform_img(image)
        return image

    def load_img(self, path):
        from PIL import Image
        import numpy as np
        import blobfile as bf

        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        return pil_image

'''

from torch.utils.data import DataLoader
import glob, os

def load_optimus(PATH_TO_CHECKPOINT = "/mnt/dataset/h_optimus_0/checkpoint.pth", device='cuda'):
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

    model_h_optimus = timm.models.VisionTransformer(**params)
    model_h_optimus.load_state_dict(torch.load(PATH_TO_CHECKPOINT, map_location="cpu"))

    model_h_optimus.eval()
    model_h_optimus.to(device)

    return model_h_optimus

'''
def get_monuseg_dataloader(data_path, version="v1.2", batch_size=1, split="", imagesize=224, resize=256, subsample=None):

    print(f"{split} dataloader : ")


    if split == "test":
        images = []
        images_syn = glob.glob(os.path.join(data_path, f"test/syn/{version}_*/samples/", "*.png"))
        images_gt = glob.glob(os.path.join(data_path, "test/gt", "*.png"))

        if subsample:
            t = 1000 * time.time() # current time in milliseconds
            np.random.seed(int(t) % 2**32)

            images_syn = list(np.random.choice(images_syn, int(len(images_syn)*subsample), replace=False))
            images_gt = list(np.random.choice(images_gt, int(len(images_gt)*subsample), replace=False))

            print(hash(''.join(images_syn)))
            print(hash(''.join(images_gt)))

        images+= images_syn
        images+= images_gt
        

    else:
        images = glob.glob(os.path.join(data_path, split, "gt", "*.png"))

    print(hash(''.join(images)))


    image_dataset = ImageDataset(images, imagesize=imagesize, resize=resize)
    dataloader = DataLoader(image_dataset, batch_size=batch_size, shuffle=False)

    print(f"\tsample : {subsample}")
    print(f"\timages : {len(images)}")

    return dataloader
'''



def get_h_optimus_feat(images, PATH_TO_CHECKPOINT = "/mnt/dataset/h_optimus_0/checkpoint.pth", model=None, batch_size=1):

    print("Loading model...")

    if model is None:

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
            'no_embed_class': True, 
            # 'img_size': 128, 
            'img_size': 224, 
            'num_classes': 0, 
            'in_chans': 3
        }

        model = timm.models.VisionTransformer(**params)

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



if __name__ == "__main__":
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
        'no_embed_class': True, 
        # 'img_size': 128, 
        'img_size': 224, 
        'num_classes': 0, 
        'in_chans': 3
    }

    model_h_optimus = timm.models.VisionTransformer(**params)
