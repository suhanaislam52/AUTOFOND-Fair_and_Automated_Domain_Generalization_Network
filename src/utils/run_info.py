from typing import Any, Dict, List, Tuple

import wandb
from tqdm import tqdm


def get_project_runs(
    api_conn: wandb.Api,
    project_name: str,
    sweep_ids: List[str],
    filtering_criteria: Dict[str, Any],
) -> Tuple[List[wandb.run], List[str]]:
    print(f"[info] Extracting sweep information from the following sweeps: {sweep_ids}")
    runs = []
    unique_datasets = []
    for sweep_id in sweep_ids:
        sweep = api_conn.sweep(f"{project_name}/{sweep_id}")
        for run in tqdm(sweep.runs):
            invalid_run = False
            # remove runs that don't match all of the filteria criteria
            for filtering_key, filtering_value in filtering_criteria.items():
                if run.config[filtering_key] != filtering_value:
                    invalid_run = True
                    break
            if not invalid_run:
                if run.config["dataset"] not in unique_datasets:
                    unique_datasets.append(run.config["dataset"])
                runs.append(run)
    print(f"[info] Runs to process: {len(runs)}")
    return runs, unique_datasets


def find_best_steps(
    runs: List[wandb.run],
    val_metric: str,
    test_metric: str,
    metric_goal: str,
):
    """
    Returns the best validation peformance, the step
    and the corresponding test performance at that step
    """
    assert metric_goal in ["min", "max"]

    best_step_info = {}
    # for each run
    for run in tqdm(runs):
        # get the validation history
        run_validation_history = run.scan_history(keys=[val_metric, "step"])
        run_test_history = run.scan_history(keys=[test_metric, "step"])

        best_metric = float("inf") if metric_goal == "min" else float("-inf")
        best_step = None

        # identify the best validation performance and step
        for row in run_validation_history:
            step_metric = row[val_metric]
            if metric_goal == "min" and step_metric < best_metric:
                best_metric = step_metric
                best_step = row["step"]
            elif metric_goal == "max" and step_metric > best_metric:
                best_metric = step_metric
                best_step = row["step"]

        # grab the correpsonding validation performance
        test_value = None
        for row in run_test_history:
            step = row['step']
            if step == best_step:
                test_value = row[test_metric]
        assert test_value is not None, 'Could not find the corresponding test_value'

        best_step_info[run] = {
            val_metric: best_metric,
            "step_number": best_step,
            test_metric: test_value
        }
    return best_step_info
