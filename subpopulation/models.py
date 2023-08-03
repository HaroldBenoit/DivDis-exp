import torch.nn as nn
import torch
import torchvision
from resnet_simclr import get_resnet
from robust_resnet import get_robust_resnet50
from typing import List


model_attributes = {
    "bert": {"feature_type": "text"},
    "distilbert": {"feature_type": "text"},
    "inception_v3": {
        "feature_type": "image",
        "target_resolution": (299, 299),
        "flatten": False,
    },
    "vit_b_16": {
        "feature_type": "image",
        "target_resolution": (224, 224),
        "flatten": False,
    },
    "vit_b_16_resnet50": {
        "feature_type": "image",
        "target_resolution": (224, 224),
        "flatten": False,
    },
        "vit_b_16_resnet50_np": {
        "feature_type": "image",
        "target_resolution": (224, 224),
        "flatten": False,
    },
        "resnet50_resnet50_np": {
        "feature_type": "image",
        "target_resolution": (224, 224),
        "flatten": False,
    },
    "wideresnet50": {
        "feature_type": "image",
        "target_resolution": (224, 224),
        "flatten": False,
    },
    "resnet50": {
        "feature_type": "image",
        "target_resolution": (224, 224),
        "flatten": False,
    },
    "resnet50SIMCLRv2": {
        "feature_type": "image",
        "target_resolution": (224, 224),
        "flatten": False,
    },
    "resnet50SwAV": {
        "feature_type": "image",
        "target_resolution": (224, 224),
        "flatten": False,
    },
    "resnet50MocoV2": {
        "feature_type": "image",
        "target_resolution": (224, 224),
        "flatten": False,
    },
    "robust_resnet50": {
        "feature_type": "image",
        "target_resolution": (224, 224),
        "flatten": False,
    },
    "resnet34": {"feature_type": "image", "target_resolution": None, "flatten": False},
    "raw_logistic_regression": {
        "feature_type": "image",
        "target_resolution": None,
        "flatten": True,
    },
    "densenet121": {
        "feature_type": "image",
        "target_resolution": (224, 224),
        "flatten": False,
    },
}




class ConcatenatedModel(nn.Module):
    def __init__(self, models:nn.ModuleList):

        super(ConcatenatedModel, self).__init__()
        self.models = models

    def forward(self, x):

        # Concatenate the outputs
        concatenated_output = torch.cat([model(x) for model in self.models], dim=1)

        return concatenated_output




def get_model(model_name, args, pretrained, n_classes):

    if model_name == "resnet50":
        model = torchvision.models.resnet50(pretrained=pretrained)
        d = model.fc.in_features
        if args.head_only:
            for p in model.parameters():
                p.requires_grad = False
        model.fc = nn.Linear(d, n_classes)
    elif model_name == "vit_b_16":
        pretrained_str = None if not(pretrained) else 'IMAGENET1K_V1'
        model = torchvision.models.vit_b_16(weights=pretrained_str)
        model.heads = nn.Linear(in_features=768, out_features=n_classes, bias=True)

    elif model_name == "resnet50_np":
        model = torchvision.models.resnet50(pretrained=False)
        d = model.fc.in_features
        if args.head_only:
            for p in model.parameters():
                p.requires_grad = False
        model.fc = nn.Linear(d, n_classes)
    
    elif model_name == "vit_b_16_np":
        pretrained_str = None 
        model = torchvision.models.vit_b_16(weights=pretrained_str)
        model.heads = nn.Linear(in_features=768, out_features=n_classes, bias=True)

    elif model_name == "resnet50SIMCLRv2":
        model, _ = get_resnet(depth=50, width_multiplier=1, sk_ratio=0)
        state = torch.load("/datasets/home/hbenoit/SimCLRv2-Pytorch/r50_1x_sk0.pth")
        model.load_state_dict(state["resnet"])
        d = model.fc.in_features
        if args.head_only:
            for p in model.parameters():
                p.requires_grad = False
        model.fc = nn.Linear(d, n_classes)
    elif model_name == "resnet50SwAV":
        model = torch.hub.load('facebookresearch/swav:main', 'resnet50')
        d = model.fc.in_features
        if args.head_only:
            for p in model.parameters():
                p.requires_grad = False
        model.fc = nn.Linear(d, n_classes)
    elif model_name == "resnet50MocoV2":
        model = torchvision.models.resnet50(pretrained=pretrained)
        d = model.fc.in_features

        state = torch.load("/datasets/home/hbenoit/mocov2/moco_v2_800ep_pretrain.pth.tar")
        new_state = {k.replace("module.encoder_q.",""):v for k,v in state["state_dict"].items()}
        for i in ["0","2"]:
            new_state.pop(f"fc.{i}.bias")
            new_state.pop(f"fc.{i}.weight")
            
        model.load_state_dict(new_state, strict=False)

        if args.head_only:
            for p in model.parameters():
                p.requires_grad = False
        model.fc = nn.Linear(d, n_classes)

    elif model_name == "robust_resnet50":
        robust = get_robust_resnet50()
        state = torch.load("/datasets/home/hbenoit/robust_resnet/resnet50_l2_eps0.05.ckpt")
        new_state = {}
        for k in state["model"]:
            if "attacker" not in k:
                new_state [k.replace("module.","")] = state["model"][k]
        robust.load_state_dict(new_state)
        d = robust.model.fc.in_features
        if args.head_only:
            for p in robust.model.parameters():
                p.requires_grad = False
        robust.model.fc = nn.Linear(d, n_classes)

        ## assume model
        model = robust
        
    elif model_name == "resnet34":
        model = torchvision.models.resnet34(pretrained=pretrained)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    elif model_name == "wideresnet50":
        model = torchvision.models.wide_resnet50_2(pretrained=pretrained)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    elif model_name == "densenet121":
        model = torchvision.models.densenet121(pretrained=pretrained)
        d = model.classifier.in_features
        model.classifier = nn.Linear(d, n_classes)


    return model





