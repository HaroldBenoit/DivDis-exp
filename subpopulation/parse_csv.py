import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from loguru import logger
from pathlib import Path

log_file = "parsed_results.txt"
if os.path.exists(log_file):
    os.remove(log_file)
log_format = "{message}"
logger.remove()
logger.add(sys.stdout, format=log_format, level="INFO", colorize=True)
logger.add(log_file, format=log_format, level="INFO", colorize=True)


def get_vals(val_csvs, test_csvs, keys, run_slice=None, return_similarity=False):
    if not run_slice:
        run_slice = slice(len(val_csvs[keys[0]]))
    val_metrics = np.stack([val_csvs[k].values for k in keys], axis=0)[:, run_slice]
    test_metrics = np.stack([test_csvs[k].values for k in keys], axis=0)[:, run_slice]

    if return_similarity:
        val_sims = np.stack([val_csvs[k].values for k in ["similarity_mean"]], axis=0)[:, run_slice]
        test_sims = np.stack([test_csvs[k].values for k in ["similarity_mean"]], axis=0)[:, run_slice]

    # Cut to min length, prevents errors when only one of two csvs were updated
    both_N = min(val_metrics.shape[-1], test_metrics.shape[-1])
    val_metrics, test_metrics = val_metrics[:, :both_N], test_metrics[:, :both_N]

    best_idx = val_metrics.argmax()
    n_model, n_epoch = np.unravel_index(best_idx, val_metrics.shape)
    val_max = val_metrics[n_model, n_epoch]

    test_max = np.max(test_metrics)
    test_cv = test_metrics[n_model, n_epoch]
    print(f"Best epoch {n_epoch}")

    res = {"val": val_max, "test": test_max, "test_cv": test_cv}

    if return_similarity:
        val_sim = val_sims[0,n_epoch]
        test_sim = test_sims[0, n_epoch]

        res["val_sim"] = val_sim 
        res["test_sim"] = test_sim

    return res


def summarize(regex, no_sim=False):
    logger.info(f"\n\nSorting for {regex=}")
    filenames = glob.glob(regex)
    exp_settings = [
        (int(re.search("h[0-9]+", os.path.basename(fn)).group()[1:]), fn[:-9])
        for fn in filenames
    ]
    exp_settings = sorted(exp_settings, key=lambda x: x[1])

    all_exp_logs = []
    for n, exp_name in exp_settings:
        try:
            val_csv = pd.read_csv(f"{exp_name}_val.csv")
            test_csv = pd.read_csv(f"{exp_name}_test.csv")
            assert len(val_csv) >= 1 and len(test_csv) >= 1
            all_exp_logs.append((n, exp_name, (val_csv, test_csv)))
        except:
            logger.info(f"Can't read {exp_name}!")

    logger.info("\n\nStandard Performance (Avg Acc, Worst Acc):")
    worsts_dict = defaultdict(list)
    avgs_dict = defaultdict(list)
    for n, name, (val_csv, test_csv) in all_exp_logs:
        run_slice = None
        if "div" in name:
            worst_keys = [f"h{i}_worst_group_acc" for i in range(n)]
            avg_keys = [f"h{i}_group_avg_acc" for i in range(n)]
        else:
            worst_keys = ["worst_group_acc"]
            avg_keys = ["group_avg_acc"]
        #print(n, name)
        worst_accs = get_vals(val_csv, test_csv, worst_keys, run_slice=run_slice, return_similarity=not(no_sim))
        avg_accs = get_vals(val_csv, test_csv, avg_keys, run_slice=run_slice)
        
        setting_name = name[:-2]
        path = Path(setting_name)
        ## assuming seed folder
        setting_name = os.path.join(path.parents[1], setting_name.split("/")[-1])
        print("Setting", setting_name)
        worsts_dict[setting_name].append(worst_accs)
        avgs_dict[setting_name].append(avg_accs)

    keys_worst = sorted(

        worsts_dict.keys(),
        key=lambda k: np.mean([i["test_cv"] for i in worsts_dict[k]]),
        reverse=True,
    )[:25]

    logger.info(f"Regex:{regex}\n")
    logger.info("\nSorted by worst (Average acc, Worst-group acc, Test similarity, Unlabeled similarity)")
    for key in keys_worst:
        avgs = [i["test_cv"] for i in avgs_dict[key]]
        worsts = [i["test_cv"] for i in worsts_dict[key]]

        N = len(worsts)
        avg_string = f"{np.mean(avgs):.3f} +- {np.std(avgs):.3f}"
        worst_string = f"{np.mean(worsts):.3f} +- {np.std(worsts):.3f}"
        
        res_string = f"{os.path.basename(key):<80}\t{N}  {avg_string}    {worst_string}"

        if not(no_sim):
            sims = [i["test_sim"] for i in worsts_dict[key]]
            sims_string = f"{np.mean(sims):.3f} +- {np.std(sims):.3f}"
            val_sims = [i["val_sim"] for i in worsts_dict[key]]
            val_sims_string = f"{np.mean(val_sims):.3f} +- {np.std(val_sims):.3f}"
            res_string = f"{os.path.basename(key):<80}\t{N}  {avg_string}    {worst_string}    {sims_string}    {val_sims_string}"

        logger.info(
            res_string
        )

#summarize("logs/final_results_10-10/*_test.csv", no_sim=True)
#summarize("logs/final_results_100-10/*_test.csv", no_sim=True)
#summarize("paper_exp/waterbirds/logs/div*/*waterbirds/seed*/*_test.csv")

#summarize("paper_exp/waterbirds/logs/grey/div10/heads*/*waterbirds/seed*/*_test.csv")
#summarize("paper_exp/waterbirds/logs/heads*/div*/np_cc_waterbirds/seed*/*_test.csv")

#summarize("paper_exp/waterbirds/logs/simclr/div*/*waterbirds/seed*/*_test.csv")
#summarize("paper_exp/waterbirds/logs/robust/div*/*waterbirds/seed*/*_test.csv")
#summarize("paper_exp/waterbirds/logs/swav/div*/*waterbirds/seed*/*_test.csv")
#summarize("paper_exp/waterbirds/logs/moco/div*/*waterbirds/seed*/*_test.csv")
#summarize("paper_exp/waterbirds/logs/vit_b_16/div*/*waterbirds/seed*/*_test.csv")
#summarize("paper_exp/waterbirds/logs/two_models/div*/*waterbirds/seed*/*_test.csv")
#summarize("paper_exp/waterbirds/logs/resnet50_resnet50_np/div*/*waterbirds/seed*/*_test.csv")
#summarize("paper_exp/waterbirds/logs/vit_b_16_resnet50_np/div*/*waterbirds/seed*/*_test.csv")

summarize("paper_exp/waterbirds/logs/mult_models/div*/*waterbirds/seed*/*_test.csv")

#summarize("paper_exp/waterbirds/logs/inverse/div*/*waterbirds/seed*/*_test.csv")
#summarize("paper_exp/celeba/logs/np_celeba_1_cc/*_test.csv")
#summarize("paper_exp/celeba/logs/*celeba_2/seed*/*_test.csv")