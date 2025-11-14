import argparse

import wandb
import yaml

from src.train.fit import fit
from src.utils.hparams import random_hparams, seed_hash
from src.utils.run_info import get_project_runs, find_best_steps

if __name__ == "__main__":
    # Configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/extract_config.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    api = wandb.Api(
        overrides={
            "entity": config["wandb"]["entity"],
        }
    )

    sweep_ids = config["wandb"]["sweep_ids"]
    project_name = config["wandb"]["project"]
    runs, unique_datasets = get_project_runs(
        api_conn=api,
        project_name=project_name,
        sweep_ids=sweep_ids,
        filtering_criteria=config["filtering_criteria"],
    )

    metric_name = config["metric"]["name"]
    metric_goal = config["metric"]["goal"]
    best_steps = find_best_steps(
        runs=runs, metric_name=metric_name, metric_goal=metric_goal
    )

    # Optionally outputs the best metric value and step for each run in sweep
    if config["script_options"]["display_all_runs"]:
        print(f"[info] {metric_name} for each run:")
        print(best_steps)

    best_runs_output = {dataset: {} for dataset in unique_datasets}
    for run, run_step_data in best_steps.items():
        run_id = run.id
        run_dataset = run.config["dataset"]
        run_test_id = run.config["test_set_id"]
        run_metric_value = run_step_data[metric_name]

        if (
            (run_test_id not in best_runs_output[run_dataset])
            or (
                metric_goal == "min"
                and run_metric_value
                < best_runs_output[run_dataset][run_test_id][metric_name]
            )
            or (
                metric_goal == "max"
                and run_metric_value
                > best_runs_output[run_dataset][run_test_id][metric_name]
            )
        ):
            best_runs_output[run_dataset][run_test_id] = {
                "run_id": run_id,
                **run_step_data,
            }

    print(f"[info] The best runs:")
    for dataset in best_runs_output:
        print(f"Dataset {dataset}")
        for test_id in best_runs_output[dataset]:
            print(f"Test Id: {test_id}, {best_runs_output[dataset][test_id]}")
        print("")
