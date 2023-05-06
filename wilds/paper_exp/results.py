import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.insert(0, '/datasets/home/hbenoit/DivDis-exp/wilds/')
from algorithms.DivDis import DivDis, MultiHeadModel
from models.initializer import initialize_torchvision_model
from utils import load, move_to
import wilds
from wilds.common.data_loaders import get_eval_loader, get_train_loader
from transforms import initialize_transform
import json
from argparse import Namespace
import argparse
import os
import copy
from wilds.common.grouper import CombinatorialGrouper

DEVICE = "cuda:0"

class MultiHeadModel(nn.Module):
    def __init__(self, featurizer, classifier, heads=2):
        super().__init__()
        self.heads = heads
        self.featurizer = featurizer
        in_dim, out_dim = classifier.in_features, classifier.out_features * self.heads
        self.heads_classifier = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        features = self.featurizer(x)
        outputs = self.heads_classifier(features)
        return outputs
    

    def process_batch(self, batch):
        """
        Overrides single_model_algorithm.process_batch().
        Args:
            - batch (x, y, m): a batch of data yielded by data loaders
        Output:
            - results (dictionary): information about the batch
                - y_true (Tensor): ground truth labels for batch

                - y_pred (Tensor): model output for batch

        """
        # Labeled examples
        x, y_true, metadata = batch
        x = move_to(x, DEVICE)
        y_true = move_to(y_true, DEVICE)
        # package the results
        results = { "y_true": y_true, "metadata": metadata}

        pred = self.forward(x)
        preds_chunked = torch.chunk(pred, self.heads, dim=-1)
        for i in range(self.heads):
            results[f"y_pred_{i}"] = preds_chunked[i]

        return results


from tqdm import tqdm
from utils import detach_and_clone, collate_list
from configs.supported import process_outputs_functions, process_pseudolabels_functions


def multiclass_logits_to_pred(logits):
    """
    Takes multi-class logits of size (batch_size, ..., n_classes) and returns predictions
    by taking an argmax at the last dimension
    """
    assert logits.dim() > 1
    return logits.argmax(-1)

def run_test_epoch(
    algorithm, dataset,
):

    algorithm.eval()
    torch.set_grad_enabled(False)

    # Not preallocating memory is slower
    # but makes it easier to handle different types of data loaders
    # (which might not return exactly the same number of examples per epoch)
    epoch_y_true = []
    epoch_y_pred = [[] for i in range(algorithm.heads)]
    epoch_metadata = []

    # Assert that data loaders are defined for the datasets
    assert "loader" in dataset, "A data loader must be defined for the dataset."


    batches = dataset["loader"]
    batches = tqdm(batches)
    last_batch_idx = len(batches) - 1


    # Using enumerate(iterator) can sometimes leak memory in some environments (!)
    # so we manually increment batch_idx
    batch_idx = 0
    for labeled_batch in batches:

        batch_results = algorithm.process_batch(labeled_batch)

        # These tensors are already detached, but we need to clone them again
        # Otherwise they don't get garbage collected properly in some versions
        # The extra detach is just for safety
        # (they should already be detached in batch_results)
        epoch_y_true.append(detach_and_clone(batch_results["y_true"]))
        for i in range(algorithm.heads):
            preds= batch_results[f"y_pred_{i}"]
            #print(f"PREDS_{i}", preds)
            preds = multiclass_logits_to_pred(preds)
            epoch_y_pred[i].append(detach_and_clone(preds))


        epoch_metadata.append(detach_and_clone(batch_results["metadata"]))


        batch_idx += 1


    epoch_y_true = collate_list(epoch_y_true).to("cpu")
    epoch_metadata = collate_list(epoch_metadata).to("cpu")

    ## HERE IT TAKES THE EPOCH PREDICTIONS AND OUTPUTS CORRECT EVALUATION METRICS e.g. worst-group acc
    DEVICE = "cpu"
    results={}
    for i in range(algorithm.heads):
        pred = collate_list(epoch_y_pred[i]).to("cpu")
        curr_key=f"res_h_{i}"
        results[curr_key], _ = dataset["dataset"].eval(
        pred, epoch_y_true, epoch_metadata
        )
        results[f"epoch_y_pred_h_{i}"] = pred.tolist()


    results["epoch_y_true"] = epoch_y_true.tolist()
    results["epoch_metadata"] = epoch_metadata.tolist()

    return results


def main():

    parser = argparse.ArgumentParser(
        description="Evaluate predictions for WILDS datasets."
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="/datasets/home/hbenoit/DivDis-exp/wilds/logs/"
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["densenet121","resnet50"]
    )

    parser.add_argument(
        "--seeds",
        nargs="*",
        type=str,
        default=None,
        help="Specifies the seeds to be evaluated on")
    
    parser.add_argument(
        "--folder_format",
        type=str,
        default=None,
        help="Specifies the format of the folder name under which the results are stored e.g. camelyon_seed{seed}"
    )

    parser.add_argument(
        "--frac",
        type=float,
        default=1.0
    )

    args= parser.parse_args()

    log_dir = args.log_dir

    for seed in args.seeds:

        folder_path = os.path.join(log_dir, args.folder_format.format(seed=seed))
        checkpoint_path = os.path.join(folder_path, f"camelyon17_seed:{seed}_epoch:best_model.pth" )
        res =  torch.load(checkpoint_path)

        prev_epoch = res["epoch"]
        best_val_metric = res["best_val_metric"]

        print(
        f"Resuming from epoch {prev_epoch} with best val metric {best_val_metric}")

        model= res["algorithm"]

        ## correct key names
        new_model = copy.deepcopy(model)
        for key in model:
            if "model.featurizer." in key:
                new_key = key.replace("model.featurizer." ,"")
                new_model[new_key] = model[key]
                del new_model[key]
            elif "" in key:
                new_key = key.replace("model.heads_classifier.","")
                new_model[new_key] = model[key]
                del new_model[key]


        featurizer = initialize_torchvision_model(
            name=args.model, d_out=None, **{"pretrained":False}
        )
        classifier = nn.Linear(featurizer.d_out, 2)

        ## load weight
        featurizer.load_state_dict(new_model,strict=False)
        algorithm = MultiHeadModel(featurizer=featurizer, classifier=classifier)
        #algorithm.featurizer.load_state_dict(new_model,strict=False)
        print(new_model["weight"])
        print(new_model["bias"])
        algorithm.heads_classifier.load_state_dict({"weight":new_model["weight"], "bias":new_model["bias"]}, strict=True)
        


        # Data
        full_dataset = wilds.get_dataset(
                dataset="camelyon17",
                root_dir="/datasets/home/hbenoit/D-BAT-exp/datasets/",
                download=False,
                split_scheme="official",
            )

        config = Namespace(**{"target_resolution": (224, 224)})

        eval_transform = initialize_transform(transform_name="image_base", dataset=full_dataset, is_training=False, config=config)

        test_data = full_dataset.get_subset("test", transform=eval_transform, frac=args.frac)

        train_grouper = CombinatorialGrouper(
            dataset=full_dataset, groupby_fields=["hospital"]
        )

        test_loader = get_eval_loader(
            loader="standard",
            dataset=test_data,
            batch_size=32,
            grouper=train_grouper,
        )

        dataset = {"loader": test_loader, "dataset":full_dataset}

        results = run_test_epoch(algorithm=algorithm.to(DEVICE), dataset=dataset)

        h0 = results["epoch_y_pred_h_0"]
        h1 = results["epoch_y_pred_h_1"]
        h0 = torch.tensor(h0)
        h1 = torch.tensor(h1)
        sim = (h0 == h1).float().mean()

        print(f"SIMILARITY = {sim}")
        for i in range(2):
            acc = results[f"res_h_{i}"]["acc_avg"]
            print(f"TEST ACC h_{i} = {acc}")

        filename = os.path.join(folder_path, "save_preds.json")

        
        with open(filename,'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        print(f"{args.folder_format.format(seed=seed)} preds.json  succesfully created")

if __name__ == "__main__":
    main()
