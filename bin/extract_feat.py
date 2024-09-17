import sys, os

# TODO : Change absolute path
sys.path.append('/workspace/patchcore-inspection/src')
sys.path.append('/workspace/patchcore-inspection/')
from patchcore.datasets.monuseg import get_monuseg_images, MoNuSegDataset

from feature_extractor.utils import extract_features
import json, os
import numpy as np
import torch

if __name__ == "__main__":

    # backbone = "medsam"
    # backbone = "optimus_old"
    backbone = "resnet50"
    # backbone = "optimus"

    layer = "out"

    input_output_map = {
        "/mnt/dataset/MoNuSeg/patches_valid_inst_128x128_128x128/MoNuSegTrainingData/images" : f"/mnt/dataset/MoNuSeg/patches_valid_inst_128x128_128x128/MoNuSegTrainingData/images_feat_{backbone}.json",
        "/mnt/dataset/MoNuSeg/patches_valid_inst_128x128_128x128/MoNuSegTest/images" : f"/mnt/dataset/MoNuSeg/patches_valid_inst_128x128_128x128/MoNuSegTest/images_feat_{backbone}.json",
        "/mnt/dataset/MoNuSeg/out_sdm_128x128/patches_valid_128.32CH_1000st_1e-4lr_8bs_hvb_col_cos_clus6/v1.3*/images" : "/mnt/dataset/MoNuSeg/out_sdm_128x128/patches_valid_128.32CH_1000st_1e-4lr_8bs_hvb_col_cos_clus6/v1.3_feat_{backbone}.json"
    }

    for (data_dir, out_json) in input_output_map.items():

        if os.path.exists(out_json):
            print("File Exists : ", out_json)
            continue

        images = get_monuseg_images('', [data_dir], subsample=None)
        print(len(images))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        img_features = extract_features(images, backbone, device=device, layers=[layer])
        img_feature_out = img_features[layer].squeeze()

        assert img_feature_out.ndim == 2, f"Two dimensions expected but img_feature_out has dimension {img_feature_out.ndim}"

        with open(out_json, 'w+') as f:
            json.dump(img_feature_out, f)
        



# embedding_path = os.path.join('./logs_clustered/MoNuSeg/patches_valid_inst_128x128_128x128', f"{PREPROCESS}_{EMB_MODEL_NAME}")

features_dict = {}

for k, v in images.items():
    feat_path = k + f"_feat_{backbone}.json"
    if os.path.exists(feat_path):
        with open(feat_path, 'r') as f:
            feat_json = json.load(f)

            opt_feat_flat = [feat_json[i] for i in v]
            opt_feat_flat = np.array(list(opt_feat_flat))  

            print(feat_path, opt_feat_flat.shape)        

            features_dict[k] = opt_feat_flat

    else:
        raise NotImplementedError








