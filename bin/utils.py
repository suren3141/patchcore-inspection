def get_backbone(backbone_name, backbone_seed):

    if backbone_name == "optimus":
        from patchcore.optimus import load_optimus

        backbone = load_optimus()
        backbone.name, backbone.seed = backbone_name, backbone_seed
    if backbone_name == "medsam":
        from segment_anything import SamPredictor, sam_model_registry

        model_weights_path = "/mnt/dataset/medsam/medsam_vit_b.pth"
        sam = sam_model_registry["vit_b"](checkpoint=model_weights_path)
        # predictor = SamPredictor(sam)
        backbone = sam
        backbone.name, backbone.seed = backbone_name, backbone_seed
    else:
        import sys
        sys.path.append('src/patchcore')
        import patchcore.backbones

        backbone = patchcore.backbones.load(backbone_name)
        backbone.name, backbone.seed = backbone_name, backbone_seed
