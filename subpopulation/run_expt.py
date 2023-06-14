import os

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing
import torch.nn as nn
import torchvision

os.environ["TRANSFORMERS_CACHE"] = "/tmp/transformers"
from parser import get_args

from data import folds
from data.data import log_data, log_meta_data, prepare_data
from data.dro_dataset import DRODataset
from data.folds import Subset
from models import model_attributes, ConcatenatedModel
from train import train
from utils import CSVBatchLogger, Logger, construct_loader, log_args, set_seed
from resnet_simclr import get_resnet
from robust_resnet import get_robust_resnet50
import sys

def main():
    args = get_args()

    env_variables = os.environ
    for k in ["SLURM_JOB_NODELIST", "SHELL", "PWD"]:
        if k in env_variables.keys():
            print(f"{k:20s}: {env_variables[k]}")

    if os.path.exists(args.log_dir) and args.resume:
        resume = True
        mode = "a"
    else:
        resume = False
        mode = "w"

    ## Initialize logs
    os.makedirs(args.log_dir, exist_ok=True)

    logger = Logger(os.path.join(args.log_dir, f"{args.exp_string}_log.txt"), mode)

    # Record args
    log_args(args, logger)
    print(args.exp_string)

    set_seed(args.seed)

    print("SET SEED")

    # Data
    # Test data for label_shift_step is not implemented yet
    test_data = None
    test_loader = None

    print("prepare data")
    if args.shift_type == "confounder":
        train_data, val_data, test_data = prepare_data(args, train=True)
    elif args.shift_type == "label_shift_step":
        train_data, val_data = prepare_data(args, train=True)
    else:
        raise NotImplementedError

    train_data.n_groups = len(np.unique(train_data.get_group_array()))
    val_data.n_groups = len(np.unique(val_data.get_group_array()))
    test_data.n_groups = len(np.unique(test_data.get_group_array()))

    assert not args.fold or not args.up_weight

    if args.fold:
        train_data, val_data = folds.get_fold(
            train_data,
            args.fold,
            cross_validation_ratio=(1 / args.num_folds_per_sweep),
            num_valid_per_point=args.num_sweeps,
            seed=args.seed,
        )

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": 4,
        "pin_memory": True,
    }

    ## modify val data to make it only have minority groups
    if args.inverse_correlation:
        if args.dataset == "CUB":
            minority_groups = (1, 2)
        else:
            raise NotImplementedError("no inverse correlation for this dataset")
        
        minority_idxs = [
            np.where(val_data.get_group_array() == i)[0] for i in minority_groups
        ]
        minority_idxs = np.concatenate(minority_idxs)
        temp_val_data = DRODataset(
            Subset(val_data, minority_idxs),
            process_item_fn=None,
            n_groups=val_data.n_groups,
            n_classes=val_data.n_classes,
            group_str_fn=val_data.group_str,
        )
        val_loader = temp_val_data.get_loader(
            train=False, reweight_groups=None, **loader_kwargs
        )
        _, counts = np.unique(val_data.get_group_array(), return_counts=True)
        print(
            f"Using minority classes only for unlabeled {minority_groups}. Counts: {counts}. Selected {len(temp_val_data)} of {len(val_data)} datapoints."
        )

    else:
        val_loader = construct_loader(
            val_data, train=False, reweight_groups=None, loader_kwargs=loader_kwargs
        )


    test_loader = construct_loader(
        test_data, train=False, reweight_groups=None, loader_kwargs=loader_kwargs
    )

    print("Get loader")
    data = {}
    if args.majority_only:
        _, counts = np.unique(train_data.get_group_array(), return_counts=True)
        if args.dataset == "CUB":
            majority_groups = (0, 3)
        elif args.dataset in ["CelebA", "CMNIST"]:
            majority_groups = (1, 2)
        elif args.dataset == "MultiNLI":
            if args.majority_setting == "02345":
                majority_groups = (0, 2, 3, 4, 5)
            elif args.majority_setting == "0124":
                majority_groups = (0, 1, 2, 4)
        else:
            raise ValueError(
                f"Majority-only loader is not implemented for {args.dataset=}!"
            )
        majority_idxs = [
            np.where(train_data.get_group_array() == i)[0] for i in majority_groups
        ]
        majority_idxs = np.concatenate(majority_idxs)
        temp_train_data = DRODataset(
            Subset(train_data, majority_idxs),
            process_item_fn=None,
            n_groups=train_data.n_groups,
            n_classes=train_data.n_classes,
            group_str_fn=train_data.group_str,
        )
        train_loader = temp_train_data.get_loader(
            train=True, reweight_groups=False, **loader_kwargs
        )
        print(
            f"Using majority classes only {majority_groups}. Counts: {counts}. Selected {len(temp_train_data)} of {len(train_data)} datapoints."
        )
    else:
        train_loader = train_data.get_loader(
            train=True, reweight_groups=args.reweight_groups, **loader_kwargs
        )
        print("length of train_data:", len(train_data))

    data["train_data"] = train_data
    data["val_data"] = val_data
    data["test_data"] = test_data

    data["train_loader"] = train_loader
    data["val_loader"] = val_loader
    data["test_loader"] = test_loader

    if args.in_dist_testing:
        assert not args.majority_only
        all_groups, tr_counts = np.unique(
            train_data.get_group_array(), return_counts=True
        )
        tr_group_freqs = tr_counts / tr_counts.sum()

        def get_id_loader(group_freqs, dataset):
            _, counts = np.unique(dataset.get_group_array(), return_counts=True)
            id_counts = min(counts / group_freqs) * group_freqs
            group_idxs = [
                np.where(val_data.get_group_array() == i)[0] for i in all_groups
            ]
            selected_idxs = [
                np.random.choice(_idx, int(_count))
                for _count, _idx in zip(id_counts, group_idxs)
            ]
            selected_idxs = np.concatenate(selected_idxs)
            id_dataset = DRODataset(
                Subset(dataset, selected_idxs),
                process_item_fn=None,
                n_groups=val_data.n_groups,
                n_classes=val_data.n_classes,
                group_str_fn=val_data.group_str,
            )
            id_loader = id_dataset.get_loader(
                train=False, reweight_groups=None, **loader_kwargs
            )
            return id_dataset, id_loader

        val_id_dataset, val_id_loader = get_id_loader(tr_group_freqs, val_data)
        test_id_dataset, test_id_loader = get_id_loader(tr_group_freqs, test_data)
        data["val_id_data"] = val_id_dataset
        data["val_id_loader"] = val_id_loader
        data["test_id_data"] = test_id_dataset
        data["test_id_loader"] = test_id_loader

    n_classes = train_data.n_classes
    if args.diversify:
        n_classes *= args.heads

    if "Meta" in args.dataset:
        log_meta_data(data, logger)
    else:
        log_data(data, logger)

    ## Initialize model
    pretrained = not args.train_from_scratch
    if resume:
        model = torch.load(os.path.join(args.log_dir, "last_model.pth"))
        d = train_data.input_size()[0]
    elif model_attributes[args.model]["feature_type"] in (
        "precomputed",
        "raw_flattened",
    ):
        assert pretrained
        # Load precomputed features
        d = train_data.input_size()[0]
        model = nn.Linear(d, n_classes)
        model.has_aux_logits = False
    elif args.model == "resnet50":
        model = torchvision.models.resnet50(pretrained=pretrained)
        d = model.fc.in_features
        if args.head_only:
            for p in model.parameters():
                p.requires_grad = False
        model.fc = nn.Linear(d, n_classes)
    elif args.model == "vit_b_16":
        pretrained_str = None if not(pretrained) else 'IMAGENET1K_V1'
        model = torchvision.models.vit_b_16(weights=pretrained_str)
        model.heads = nn.Linear(in_features=768, out_features=n_classes, bias=True)

    elif args.model == "vit_b_16_resnet50":
        assert args.heads == 2
        pretrained_str = None if not(pretrained) else 'IMAGENET1K_V1'
        vit = torchvision.models.vit_b_16(weights=pretrained_str)
        vit.heads = nn.Linear(in_features=768, out_features=train_data.n_classes, bias=True)

        resnet = torchvision.models.resnet50(pretrained=pretrained)
        d = resnet.fc.in_features
        resnet.fc = nn.Linear(d,train_data.n_classes)
        model = ConcatenatedModel(model1=vit, model2=resnet)
    elif args.model == "vit_b_16_resnet50_np":
        assert args.heads == 2
        pretrained_str = None if not(pretrained) else 'IMAGENET1K_V1'
        vit = torchvision.models.vit_b_16(weights=pretrained_str)
        vit.heads = nn.Linear(in_features=768, out_features=train_data.n_classes, bias=True)

        resnet = torchvision.models.resnet50(pretrained=False)
        d = resnet.fc.in_features
        resnet.fc = nn.Linear(d,train_data.n_classes)
        model = ConcatenatedModel(model1=vit, model2=resnet)
    elif args.model == "resnet50_resnet50_np":
        resnet1 = torchvision.models.resnet50(pretrained=True)
        d = resnet1.fc.in_features
        resnet1.fc = nn.Linear(d,train_data.n_classes)

        resnet2 = torchvision.models.resnet50(pretrained=False)
        d = resnet2.fc.in_features
        resnet2.fc = nn.Linear(d,train_data.n_classes)
        model = ConcatenatedModel(model1=resnet1, model2=resnet2)

    elif args.model == "resnet50SIMCLRv2":
        model, _ = get_resnet(depth=50, width_multiplier=1, sk_ratio=0)
        state = torch.load("/datasets/home/hbenoit/SimCLRv2-Pytorch/r50_1x_sk0.pth")
        model.load_state_dict(state["resnet"])
        d = model.fc.in_features
        if args.head_only:
            for p in model.parameters():
                p.requires_grad = False
        model.fc = nn.Linear(d, n_classes)
    elif args.model == "resnet50SwAV":
        model = torch.hub.load('facebookresearch/swav:main', 'resnet50')
        d = model.fc.in_features
        if args.head_only:
            for p in model.parameters():
                p.requires_grad = False
        model.fc = nn.Linear(d, n_classes)
    elif args.model == "resnet50MocoV2":
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
    elif args.model == "robust_resnet50":
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
        
    elif args.model == "resnet34":
        model = torchvision.models.resnet34(pretrained=pretrained)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    elif args.model == "wideresnet50":
        model = torchvision.models.wide_resnet50_2(pretrained=pretrained)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    elif args.model == "densenet121":
        model = torchvision.models.densenet121(pretrained=pretrained)
        d = model.classifier.in_features
        model.classifier = nn.Linear(d, n_classes)

    elif "bert" in args.model:
        if args.is_featurizer:
            if args.model == "bert":
                from bert.bert import BertFeaturizer

                featurizer = BertFeaturizer.from_pretrained(
                    "bert-base-uncased", **args.model_kwargs
                )
                classifier = nn.Linear(
                    featurizer.d_out, 5 if args.dataset == "Amazon" else n_classes
                )
                model = torch.nn.Sequential(featurizer, classifier)
            elif args.model == "distilbert":
                from bert.distilbert import DistilBertFeaturizer

                featurizer = DistilBertFeaturizer.from_pretrained(
                    "distilbert-base-uncased", **args.model_kwargs
                )
                classifier = nn.Linear(
                    featurizer.d_out, 5 if args.dataset == "Amazon" else n_classes
                )
                model = torch.nn.Sequential(featurizer, classifier)
            else:
                raise NotImplementedError
        else:
            from bert.bert import BertClassifier

            model = BertClassifier.from_pretrained(
                "bert-base-uncased", num_labels=512, **args.model_kwargs
            )
    else:
        raise ValueError(f"Model {args.model} not recognized.")

    logger.flush()

    ## Define the objective
    if args.hinge:
        assert args.dataset in ["CelebA", "CUB"]  # Only supports binary

        def hinge_loss(yhat, y):
            # The torch loss takes in three arguments so we need to split yhat
            # It also expects classes in {+1.0, -1.0} whereas by default we give them in {0, 1}
            # Furthermore, if y = 1 it expects the first input to be higher instead of the second,
            # so we need to swap yhat[:, 0] and yhat[:, 1]...
            torch_loss = torch.nn.MarginRankingLoss(margin=1.0, reduction="none")
            y = (y.float() * 2.0) - 1.0
            return torch_loss(yhat[:, 1], yhat[:, 0], y)

        criterion = hinge_loss
    else:
        criterion = torch.nn.CrossEntropyLoss(reduction="none")

    if resume:
        df = pd.read_csv(os.path.join(args.log_dir, f"{args.exp_string}_test.csv"))
        epoch_offset = df.loc[len(df) - 1, "epoch"] + 1
        logger.write(f"starting from epoch {epoch_offset}")
    else:
        epoch_offset = 0

    csv_loggers = {}
    csv_loggers["train"] = CSVBatchLogger(
        args,
        os.path.join(args.log_dir, f"{args.exp_string}_train.csv"),
        train_data.n_groups,
        mode=mode,
    )
    csv_loggers["val"] = CSVBatchLogger(
        args,
        os.path.join(args.log_dir, f"{args.exp_string}_val.csv"),
        val_data.n_groups,
        mode=mode,
    )
    csv_loggers["test"] = CSVBatchLogger(
        args,
        os.path.join(args.log_dir, f"{args.exp_string}_test.csv"),
        test_data.n_groups,
        mode=mode,
    )
    if args.in_dist_testing:
        csv_loggers["val_id"] = CSVBatchLogger(
            args,
            os.path.join(args.log_dir, f"{args.exp_string}_val_id.csv"),
            val_data.n_groups,
            mode=mode,
        )
        csv_loggers["test_id"] = CSVBatchLogger(
            args,
            os.path.join(args.log_dir, f"{args.exp_string}_test_id.csv"),
            test_data.n_groups,
            mode=mode,
        )

    train(
        model,
        criterion,
        data,
        logger,
        csv_loggers,
        args,
        epoch_offset=epoch_offset,
    )

    for csv_logger in csv_loggers.values():
        csv_logger.close()


if __name__ == "__main__":
    main()
