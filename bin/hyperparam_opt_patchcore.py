import ray
# ray.remote(num_gpus=1)

resources={"gpu":1}

ray.init(
    local_mode=True,
    resources=resources,
    # address='localhost:8265',
    # address='127.0.1.1:6379',
    )

from ray import train, tune
from functools import partial
from pprint import pprint

import os, sys
from pathlib import Path
file_path =  Path(__file__).absolute()
src_dir = os.path.join(file_path.parents[1], 'src')
import sys
sys.path.append(src_dir)
os.environ["PYTHONPATH"] = src_dir
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'



def test_pytorch(assertion=False):
    import torch

    if assertion: assert torch.cuda.is_available()

    print('is cuda avaiable ',torch.cuda.is_available())
    # tune.report({'metric':0})

def objective(config, default_params=None):

    os.environ["PYTHONPATH"] = src_dir
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    # print(config)

    test_pytorch()
    
    percentage = config["sampler_percentage"]
    ps = config["patchcore_patchsize"]

    params = default_params

    model_name = params["patchcore_params"]["backbone_names"][0]

    if model_name == "optimus":
        model_name = "OPT"
    elif model_name == "medsam":
        model_name = "SAM"
    elif model_name == "wideresnet50":
        model_name = "WR50"
    else:
        raise NotImplementedError

    # print(default_params)


    params["sampler_params"]["percentage"] = percentage
    params["patchcore_params"]["patchsize"] = ps
    params["run_params"]["log_group"] = f"IM224_{model_name}_P{percentage:.1e}_D1024-1024_PS-{ps}_AN-1_S0"

    out_path = os.path.join(params["run_params"]["results_path"], params["run_params"]["log_project"], params["run_params"]["log_group"])
    print(out_path)
    pprint(params)
    if os.path.exists(os.path.join(out_path, "models")):
        results = params_to_config_eval(params)
    else:
        results = params_to_config_train(params)

    return results[0]

def actor_function():
    import debugpy
    debugpy.breakpoint()
    print('xxx')

def params_to_config_train(params):

    from run_patchcore import dataset, patch_core, sampler, run

    config = params["run_params"]

    config["methods"] = []

    k, v = dataset.callback(**params["dataset_params"])
    config["methods"].append((k, v))

    k, v = sampler.callback(**params["sampler_params"])
    config["methods"].append((k, v))

    k, v = patch_core.callback(**params["patchcore_params"])
    config["methods"].append((k, v))

    results = run(**config)

    return results

def params_to_config_eval(params):

    from load_and_evaluate_patchcore import dataset, patch_core_loader, run

    out_path = os.path.join(params["run_params"]["results_path"], params["run_params"]["log_project"], params["run_params"]["log_group"])

    # Update config file from run_params
    config = dict(
        results_path = os.path.join(out_path, "eval_results"),
        gpu = params["run_params"]["gpu"],
        seed = params["run_params"]["seed"],
        save_segmentation_images = False,
        save_anomaly_scores = True,
        methods = [],
    )

    # Dataset params updated for eval
    dataset_params = {k:v for k, v in params["dataset_params"].items() if k != "train_val_split"}
    dataset_params['subsample'] = .1
    k, v = dataset.callback(**dataset_params)
    config["methods"].append((k, v))

    # Patchmore params updated for eval. Only the location of model is needed.
    patchcore_params = dict(
        patch_core_paths = [os.path.join(out_path, "models/train")],
        faiss_on_gpu = params["patchcore_params"]["faiss_on_gpu"],
        faiss_num_workers = params["patchcore_params"]["faiss_num_workers"],
    )

    k, v = patch_core_loader.callback(**patchcore_params)
    config["methods"].append((k, v))

    results = run(**config)

    return results

def test_hyperparam(default_params):

    percentage = .1
    ps = 3

    params = default_params

    params["sampler_params"]["percentage"] = percentage
    params["patchcore_params"]["patchsize"] = ps
    params["log_group"] = f"IM224_SAM_P{percentage:.1e}_D1024-1024_PS-{ps}_AN-1_S0",

    config = params_to_config(params)

    results = run(**config)
    return results


def get_default_params():

    patchcore_params = dict(
        # -b wideresnet50 -le layer2 -le layer3
        # -b medsam -le out
        backbone_names=["wideresnet50"],
        layers_to_extract_from=["layer2", "layer3"],
        # Parameters for Glue-code (to merge different parts of the pipeline.
        pretrain_embed_dimension=1024,
        target_embed_dimension=1024,
        preprocessing="mean",
        aggregation="mean",
        # Nearest-Neighbour Anomaly Scorer parameters.
        anomaly_scorer_num_nn=5,
        # Patch-parameters.
        patchsize=3,
        patchscore=max,
        patchoverlap=0.0,
        patchsize_aggregate=[],
        # NN on GPU.
        faiss_on_gpu=True,
        faiss_num_workers=8,
    )

    dataset_params = dict(
        name = "monuseg",
        data_path = "/workspace/patchcore-inspection/monuseg",
        subdatasets = ['v1.3'],
        train_val_split = 1,
        batch_size = 2,
        num_workers = 8,
        resize = 224,
        imagesize = 0,
        augment = True,
    )

    sampler_params = dict(
        name = "approx_greedy_coreset",
        percentage = .1,
    )

    run_params = dict(
        results_path = "/mnt/dataset/patchcore/",
        gpu = [0],
        seed = 0,
        # log_group = "IM224_SAM_P01_D1024-1024_PS-5_ST-2_AN-1_S0",
        log_group = "temp",
        log_project = "monuseg_hyperparam_v1.2",
        save_segmentation_images = False,
        save_patchcore_model = True,
        methods = [],
    )

    all_params = dict(
        patchcore_params = patchcore_params,
        dataset_params = dataset_params,
        sampler_params = sampler_params,
        run_params = run_params,
    )

    return all_params


if __name__ == "__main__":

    default_params = get_default_params()

    # test_hyperparam(default_params)

    search_space = dict(
        sampler_percentage = tune.grid_search([1e-4, 1e-3, 1e-2]),
        patchcore_patchsize = tune.choice([5, 11, 19]),
        # patchcore_patchsize = tune.choice([7]),
    )


    assert ray.is_initialized()

    objective_fn = partial(objective, default_params=default_params)

    # resources={"gpu":.5}

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(objective_fn),
            resources=resources,
        ),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            max_concurrent_trials=1,
            num_samples=10,
        ),
        run_config=train.RunConfig(
                # storage_path=os.path.expanduser("~/ray_results"),
                # name="trial_fault_tolerance",
                failure_config=train.FailureConfig(max_failures=10),
        ),
    )


    results = tuner.fit()
    print(results.get_best_result(metric="instance_auroc", mode="max").config)

    ray.shutdown()
    assert not ray.is_initialized()
