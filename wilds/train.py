import math

import torch
from tqdm import tqdm
import json 
from configs.supported import process_outputs_functions, process_pseudolabels_functions
from utils import (
    InfiniteDataIterator,
    collate_list,
    detach_and_clone,
    get_model_prefix,
    get_pred_prefix,
    save_model,
    save_pred,
)
from collections import defaultdict

def run_epoch(
    algorithm, dataset, general_logger, epoch, config, train, unlabeled_dataset=None, split=None
):
    if dataset["verbose"]:
        general_logger.write(f"\n{dataset['name']}:\n")

    if train:
        algorithm.train()
        torch.set_grad_enabled(True)
    else:
        algorithm.eval()
        torch.set_grad_enabled(False)

    # Not preallocating memory is slower
    # but makes it easier to handle different types of data loaders
    # (which might not return exactly the same number of examples per epoch)
    epoch_y_true = []
    epoch_y_pred = []
    epoch_y_pred_heads = defaultdict(list)
    epoch_unlabeled_y_pred_heads = defaultdict(list)
    epoch_metadata = []

    # Assert that data loaders are defined for the datasets
    assert "loader" in dataset, "A data loader must be defined for the dataset."
    if unlabeled_dataset:
        assert (
            "loader" in unlabeled_dataset
        ), "A data loader must be defined for the dataset."

    batches = dataset["loader"]
    if config.progress_bar:
        batches = tqdm(batches)
    last_batch_idx = len(batches) - 1

    if unlabeled_dataset:
        unlabeled_data_iterator = InfiniteDataIterator(unlabeled_dataset["loader"])

    # Using enumerate(iterator) can sometimes leak memory in some environments (!)
    # so we manually increment batch_idx
    batch_idx = 0
    for labeled_batch in batches:
        if train:
            if unlabeled_dataset:
                unlabeled_batch = next(unlabeled_data_iterator)
                batch_results = algorithm.update(
                    labeled_batch,
                    unlabeled_batch,
                    is_epoch_end=(batch_idx == last_batch_idx),
                )
            else:
                batch_results = algorithm.update(
                    labeled_batch, is_epoch_end=(batch_idx == last_batch_idx)
                )
        else:

             ## individual head preds for DivDis
            if config.algorithm == "DivDis":
                ## not defined
                #if "unlabeled_data_iterator" not in globals():
                #    unlabeled_data_iterator = InfiniteDataIterator(unlabeled_dataset["loader"])
              
                unlabeled_batch = None # next(unlabeled_data_iterator)
                batch_results = algorithm.evaluate(labeled_batch, unlabeled_batch=unlabeled_batch)
                for key in batch_results.keys():
                        if "unlabeled_y_pred_" in key:
                            head_number = key[17:] ## get head number
                            y_pred = detach_and_clone(batch_results[key])
                            if config.process_outputs_function is not None:
                                y_pred = process_outputs_functions[config.process_outputs_function](y_pred)
                            epoch_unlabeled_y_pred_heads[f"unlabeled_h_{head_number}"].append(y_pred)                            
                        elif "y_pred_" in key:
                            head_number = key[7:] ## get head number
                            y_pred = detach_and_clone(batch_results[key])
                            if config.process_outputs_function is not None:
                                y_pred = process_outputs_functions[config.process_outputs_function](y_pred)
                            epoch_y_pred_heads[f"h_{head_number}"].append(y_pred)
            else:

                batch_results = algorithm.evaluate(labeled_batch)


        # These tensors are already detached, but we need to clone them again
        # Otherwise they don't get garbage collected properly in some versions
        # The extra detach is just for safety
        # (they should already be detached in batch_results)
        epoch_y_true.append(detach_and_clone(batch_results["y_true"]))
        y_pred = detach_and_clone(batch_results["y_pred"])
        if config.process_outputs_function is not None:
            y_pred = process_outputs_functions[config.process_outputs_function](y_pred)
        epoch_y_pred.append(y_pred)
        epoch_metadata.append(detach_and_clone(batch_results["metadata"]))

        if train:
            effective_batch_idx = (batch_idx + 1) / config.gradient_accumulation_steps
        else:
            effective_batch_idx = batch_idx + 1

        if train and effective_batch_idx % config.log_every == 0:
            log_results(
                algorithm,
                dataset,
                general_logger,
                epoch,
                math.ceil(effective_batch_idx),
            )

        batch_idx += 1

    epoch_y_pred = collate_list(epoch_y_pred)
    epoch_y_true = collate_list(epoch_y_true)
    epoch_metadata = collate_list(epoch_metadata)

    ## HERE IT TAKES THE EPOCH PREDICTIONS AND OUTPUTS CORRECT EVALUATION METRICS e.g. worst-group acc
    results, results_str = dataset["dataset"].eval(
        epoch_y_pred, epoch_y_true, epoch_metadata
    )

    if config.scheduler_metric_split == dataset["split"]:
        algorithm.step_schedulers(
            is_epoch=True, metrics=results, log_access=(not train)
        )

    # log after updating the scheduler in case it needs to access the internal logs
    log_results(
        algorithm, dataset, general_logger, epoch, math.ceil(effective_batch_idx)
    )

    if config.algorithm == "DivDis" and not(train):
            aux_results = {}
            for key in epoch_y_pred_heads:
                curr_key = f"res_{key}"
                pred = collate_list(epoch_y_pred_heads[key])
                aux_results[curr_key] = dataset["dataset"].eval(
                pred, epoch_y_true, epoch_metadata
                )[0]
                aux_results[f"epoch_y_pred_{curr_key}"] = pred.tolist()

            for key in epoch_unlabeled_y_pred_heads:
                curr_key = f"res_{key}"
                pred = collate_list(epoch_unlabeled_y_pred_heads[key])
                aux_results[f"epoch_unlabeled_y_pred_{curr_key}"] = pred.tolist()

            aux_results["epoch_y_true"] = epoch_y_true.tolist()
            aux_results["epoch_metadata"] = epoch_metadata.tolist()

            results["aux_results"] = aux_results

    results["epoch"] = epoch
    dataset["eval_logger"].log(results)
    if dataset["verbose"]:
        general_logger.write("Epoch eval:\n")
        general_logger.write(results_str)

    return results, epoch_y_pred


def train(
    algorithm,
    datasets,
    general_logger,
    config,
    epoch_offset,
    best_val_metric,
    unlabeled_dataset=None,
):
    """
    Train loop that, each epoch:
        - Steps an algorithm on the datasets['train'] split and the unlabeled split
        - Evaluates the algorithm on the datasets['val'] split
        - Saves models / preds with frequency according to the configs
        - Evaluates on any other specified splits in the configs
    Assumes that the datasets dict contains labeled data.
    """
    for epoch in range(epoch_offset, config.n_epochs):
        general_logger.write("\nEpoch [%d]:\n" % epoch)

        # First run training
        run_epoch(
            algorithm,
            datasets["train"],
            general_logger,
            epoch,
            config,
            train=True,
            unlabeled_dataset=unlabeled_dataset,
        )

        # Then run val
        if epoch % config.eval_every_n_epoch == 0:
            print("HERE EVAL")
            val_results, y_pred = run_epoch(
                algorithm, datasets["val"], general_logger, epoch, config, train=False, split="val"
            )
            curr_val_metric = val_results[config.val_metric]
            general_logger.write(f"Validation {config.val_metric}: {curr_val_metric:.3f}\n")

            if best_val_metric is None:
                is_best = True
            else:
                if config.val_metric_decreasing:
                    is_best = curr_val_metric < best_val_metric
                else:
                    is_best = curr_val_metric > best_val_metric
            if is_best:
                best_val_metric = curr_val_metric
                general_logger.write(
                    f"Epoch {epoch} has the best validation performance so far.\n"
                )

            ## saving divdis results
            if "aux_results" in val_results:
                aux_results = val_results["aux_results"]
                save_aux_results_if_needed(dataset=datasets["val"], aux_results=aux_results, is_best=is_best, config=config)
                  


            save_model_if_needed(
                algorithm, datasets["val"], epoch, config, is_best, best_val_metric
            )
            save_pred_if_needed(y_pred, datasets["val"], epoch, config, is_best)

            # Then run everything else
            if config.evaluate_all_splits:
                additional_splits = [
                    split for split in datasets.keys() if split not in ["train", "val"]
                ]
            else:
                additional_splits = config.eval_splits
            for split in additional_splits:
                split_results, y_pred = run_epoch(
                    algorithm, datasets[split], general_logger, epoch, config, train=False, split=split
                )
                save_pred_if_needed(y_pred, datasets[split], epoch, config, is_best)

                ## saving divdis results
                if "aux_results" in split_results:
                    aux_results = split_results["aux_results"]
                    save_aux_results_if_needed(dataset=datasets[split], aux_results=aux_results, is_best=is_best, config=config)

        general_logger.write("\n")


def evaluate(algorithm, datasets, epoch, general_logger, config, is_best):
    algorithm.eval()
    torch.set_grad_enabled(False)
    for split, dataset in datasets.items():
        if (not config.evaluate_all_splits) and (split not in config.eval_splits):
            continue
        epoch_y_true = []
        epoch_y_pred = []
        epoch_metadata = []
        iterator = tqdm(dataset["loader"]) if config.progress_bar else dataset["loader"]
        for batch in iterator:
            batch_results = algorithm.evaluate(batch)
           
            epoch_y_true.append(detach_and_clone(batch_results["y_true"]))
            y_pred = detach_and_clone(batch_results["y_pred"])
            if config.process_outputs_function is not None:
                y_pred = process_outputs_functions[config.process_outputs_function](
                    y_pred
                )
            epoch_y_pred.append(y_pred)
            epoch_metadata.append(detach_and_clone(batch_results["metadata"]))

        epoch_y_pred = collate_list(epoch_y_pred)
        epoch_y_true = collate_list(epoch_y_true)
        epoch_metadata = collate_list(epoch_metadata)
        results, results_str = dataset["dataset"].eval(
            epoch_y_pred, epoch_y_true, epoch_metadata
        )

        results["epoch"] = epoch
        dataset["eval_logger"].log(results)
        general_logger.write(f"Eval split {split} at epoch {epoch}:\n")
        general_logger.write(results_str)

        # Skip saving train preds, since the train loader generally shuffles the data
        if split != "train":
            save_pred_if_needed(
                epoch_y_pred, dataset, epoch, config, is_best, force_save=True
            )


def infer_predictions(model, loader, config):
    """
    Simple inference loop that performs inference using a model (not algorithm) and returns model outputs.
    Compatible with both labeled and unlabeled WILDS datasets.
    """
    model.eval()
    y_pred = []
    iterator = tqdm(loader) if config.progress_bar else loader
    for batch in iterator:
        x = batch[0]
        x = x.to(config.device)
        with torch.no_grad():
            output = model(x)
            if (
                not config.soft_pseudolabels
                and config.process_pseudolabels_function is not None
            ):
                _, output, _, _ = process_pseudolabels_functions[
                    config.process_pseudolabels_function
                ](
                    output,
                    confidence_threshold=config.self_training_threshold
                    if config.dataset == "globalwheat"
                    else 0,
                )
            elif config.soft_pseudolabels:
                output = torch.nn.functional.softmax(output, dim=1)
        if isinstance(output, list):
            y_pred.extend(detach_and_clone(output))
        else:
            y_pred.append(detach_and_clone(output))

    return torch.cat(y_pred, 0) if torch.is_tensor(y_pred[0]) else y_pred


def log_results(algorithm, dataset, general_logger, epoch, effective_batch_idx):
    if algorithm.has_log:
        log = algorithm.get_log()
        log["epoch"] = epoch
        log["batch"] = effective_batch_idx
        dataset["algo_logger"].log(log)
        if dataset["verbose"]:
            general_logger.write(algorithm.get_pretty_log_str())
        algorithm.reset_log()


def save_pred_if_needed(y_pred, dataset, epoch, config, is_best, force_save=False):
    if config.save_pred:
        prefix = get_pred_prefix(dataset, config)
        if force_save or (
            config.save_step is not None and (epoch + 1) % config.save_step == 0
        ):
            save_pred(y_pred, prefix + f"epoch:{epoch}_pred")
        if (not force_save) and config.save_last:
            save_pred(y_pred, prefix + f"epoch:last_pred")
        if config.save_best and is_best:
            save_pred(y_pred, prefix + f"epoch:best_pred")


def save_model_if_needed(algorithm, dataset, epoch, config, is_best, best_val_metric):
    prefix = get_model_prefix(dataset, config)
    if config.save_step is not None and (epoch + 1) % config.save_step == 0:
        save_model(
            algorithm, epoch, best_val_metric, prefix + f"epoch:{epoch}_model.pth"
        )
    if config.save_last:
        save_model(algorithm, epoch, best_val_metric, prefix + "epoch:last_model.pth")
    if config.save_best and is_best:
        save_model(algorithm, epoch, best_val_metric, prefix + "epoch:best_model.pth")

def save_aux_results_if_needed(dataset, aux_results, is_best, config):
    prefix = get_pred_prefix(dataset=dataset,config=config)

    if config.save_best and is_best:
        filename = prefix + f"epoch:best_preds.json"
        with open(filename,'w', encoding='utf-8') as f:
            json.dump(aux_results, f, ensure_ascii=False, indent=4)

    if config.save_last:
        filename = prefix + f"epoch:last_preds.json"
        with open(filename,'w', encoding='utf-8') as f:
            json.dump(aux_results, f, ensure_ascii=False, indent=4)

