"""
Simplified training loop for AutoFOND development
- No W&B dependency
- Epoch-based training (not steps)
- Simple CSV logging
- Easy to extend for auto-augmentation
"""
import collections
import logging
import os
import time
from typing import Dict, Optional

import lightning as L
import numpy as np
import torch
from tqdm import tqdm

from src.datasets import DATASETS
from src.networks import ALGORITHMS
from src.utils import misc
from src.utils.hparams import default_hparams, random_hparams


def fit_simple(
    exp_dir: str,
    logger,  # CSVLogger or PrintLogger
    seed: int,
    trial_seed: int,
    hparams_seed: int,
    algorithm_name: str,
    dataset_name: str,
    data_dir: str,
    num_workers: int,
    test_envs: list,
    overlap_type: str,
    holdout_fraction: float = 0.2,
    n_epochs: int = 100,  # Changed from n_steps!
    checkpoint_freq: int = 5,  # Now in epochs
    model_checkpoint: Optional[Dict] = None,
    teacher_paths: Optional[Dict] = None,
    num_domain_linked_classes: Optional[int] = None,
    num_classes: Optional[int] = None,
    auto_augment: bool = False,
    augment_search_epochs: int = 10,
):
    """
    Simplified training function with epoch-based training
    
    Args:
        n_epochs: Number of training epochs (replaces n_steps)
        checkpoint_freq: Evaluation frequency in epochs (replaces step-based)
        auto_augment: Enable auto-augmentation search
        augment_search_epochs: Number of epochs for augmentation policy search
    """
    # Seed everything
    L.seed_everything(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Create experiment directory
    os.makedirs(exp_dir, exist_ok=True)

    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # Setup hyperparameters
    if hparams_seed == 0:
        hparams = default_hparams(algorithm_name, dataset_name)
    else:
        hparams = random_hparams(
            algorithm_name, dataset_name, misc.seed_hash(hparams_seed, trial_seed)
        )
    logging.info(f"Hyperparameters: {hparams}")
    
    # Save hparams
    logger.save_config({"hparams": hparams}, os.path.join(exp_dir, "hparams.json"))

    # Load dataset
    dataset = DATASETS[dataset_name](
        root=data_dir,
        test_envs=test_envs,
        hparams=hparams,
        overlap_type=overlap_type,
        num_classes=num_classes,
        num_domain_linked_classes=num_domain_linked_classes,
    )
    hparams["C_oc"] = dataset.overlapping_classes
    logging.info(f"Loaded {dataset_name}")

    # Split datasets
    in_splits = []
    out_splits = []
    relative_test_env = None
    
    for env_i, env in enumerate(dataset):
        out, in_ = misc.split_dataset(
            env, int(len(env) * holdout_fraction), misc.seed_hash(trial_seed, env_i)
        )

        if hparams["class_balanced"]:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
        else:
            in_weights, out_weights = None, None

        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))

        if env.is_test_env:
            relative_test_env = env_i

    assert relative_test_env is not None, "No testing domains"
    logging.info(f"Test environment: {relative_test_env}")

    # Setup data loaders
    train_loaders = [
        misc.InfiniteDataLoader(
            dataset=env,
            weights=env_weights,
            batch_size=hparams["batch_size"],
            num_workers=num_workers,
        )
        for i, (env, env_weights) in enumerate(in_splits)
        if i != relative_test_env
    ]

    eval_loaders = [
        misc.FastDataLoader(
            dataset=env,
            batch_size=hparams["batch_size"],
            num_workers=num_workers,
        )
        for env, _ in (in_splits + out_splits)
    ]

    eval_weights = [None for _, weights in (in_splits + out_splits)]
    eval_loader_names = ["env{}_in".format(i) for i in range(len(in_splits))]
    eval_loader_names += ["env{}_out".format(i) for i in range(len(out_splits))]

    # Calculate steps per epoch
    steps_per_epoch = int(min([len(env) / hparams["batch_size"] for env, _ in in_splits if len(env) > 0]))
    total_steps = n_epochs * steps_per_epoch
    logging.info(f"Training for {n_epochs} epochs ({steps_per_epoch} steps/epoch, {total_steps} total steps)")

    # Setup algorithm
    if "distillation" in algorithm_name.lower():
        assert len(test_envs) == 1
        teacher_path = teacher_paths[dataset_name][str(test_envs[0])]
        teacher_algorithm = torch.load(teacher_path, map_location=device)
        algorithm = ALGORITHMS[algorithm_name](
            input_shape=dataset.input_shape,
            num_classes=dataset.num_classes,
            num_domains=len(dataset) - len(test_envs),
            hparams=hparams,
            teacher=teacher_algorithm,
        )
    else:
        algorithm = ALGORITHMS[algorithm_name](
            input_shape=dataset.input_shape,
            num_classes=dataset.num_classes,
            num_domains=len(dataset) - len(test_envs),
            hparams=hparams,
        )
    algorithm.to(device)
    logging.info(f"Algorithm {algorithm_name} initialized")

    # Setup model checkpointing
    if model_checkpoint is None:
        model_checkpoint = {"metric": "val/acc", "maximize": True}
    
    best_metric_value = None
    best_model_path = os.path.join(exp_dir, "best_model.ckpt")
    checkpoint_metric_parts = model_checkpoint["metric"].split("/")
    checkpoint_stage, checkpoint_metric = checkpoint_metric_parts[0], checkpoint_metric_parts[1]
    maximize = model_checkpoint["maximize"]

    # Training loop
    logging.info("Starting training loop...")
    train_minibatches_iterator = zip(*train_loaders)
    global_step = 0
    
    for epoch in range(n_epochs):
        epoch_start_time = time.time()
        epoch_metrics = collections.defaultdict(list)
        
        # Training for one epoch
        for step_in_epoch in tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{n_epochs}"):
            step_start_time = time.time()
            
            # Get batches
            minibatches_device = [
                (x.to(device), y.to(device)) 
                for x, y in next(train_minibatches_iterator)
            ]
            
            # Perform update
            step_vals = algorithm.update(minibatches_device, None)
            for key, val in step_vals.items():
                epoch_metrics[key].append(val)
            
            global_step += 1
        
        # Log epoch training metrics
        train_metrics = {f"train/{k}": np.mean(v) for k, v in epoch_metrics.items()}
        epoch_time = time.time() - epoch_start_time
        train_metrics["epoch_time"] = epoch_time
        
        # Evaluation at checkpoint frequency
        should_eval = (epoch % checkpoint_freq == 0) or (epoch == n_epochs - 1)
        
        if should_eval:
            eval_results = evaluate(
                algorithm=algorithm,
                eval_loaders=eval_loaders,
                eval_loader_names=eval_loader_names,
                eval_weights=eval_weights,
                device=device,
                dataset=dataset,
                relative_test_env=relative_test_env,
            )
            
            # Combine train and eval metrics
            all_metrics = {**train_metrics, **eval_results, "epoch": epoch, "step": global_step}
            logger.log(all_metrics)
            
            # Print summary
            print(f"\nEpoch {epoch+1}/{n_epochs}:")
            print(f"  Train Loss: {train_metrics.get('train/loss', 'N/A'):.4f}")
            print(f"  Val Acc: {eval_results.get('val/acc', 'N/A'):.4f}")
            print(f"  Test Acc: {eval_results.get('test/acc', 'N/A'):.4f}")
            
            # Save best model
            current_value = eval_results.get(f"{checkpoint_stage}/{checkpoint_metric}")
            if current_value is not None:
                is_best = (best_metric_value is None or 
                          (maximize and current_value > best_metric_value) or
                          (not maximize and current_value < best_metric_value))
                
                if is_best:
                    torch.save(algorithm, best_model_path)
                    best_metric_value = current_value
                    logging.info(f"✓ New best {model_checkpoint['metric']}: {current_value:.4f}")
        else:
            # Log only training metrics
            logger.log({**train_metrics, "epoch": epoch, "step": global_step})
    
    logging.info(f"Training complete! Best {model_checkpoint['metric']}: {best_metric_value:.4f}")
    logger.close()


def evaluate(
    algorithm,
    eval_loaders,
    eval_loader_names,
    eval_weights,
    device,
    dataset,
    relative_test_env,
):
    """Run evaluation on all data splits"""
    results = {
        stage: {metric: [] for metric in ["acc", "f1", "nacc", "oacc", "recall"]}
        for stage in ["train", "val", "test", "other"]
    }
    
    for name, loader, weights in zip(eval_loader_names, eval_loaders, eval_weights):
        acc, recall, f1, oacc, nacc, per_class_acc = misc.accuracy(
            algorithm, loader, weights, device, dataset
        )
        
        # Determine loader type
        domain_idx = int(name[3])
        if domain_idx == relative_test_env:
            loader_type = "test" if "in" in name else "other"
        elif "out" in name:
            loader_type = "val"
        else:
            loader_type = "train"
        
        # Accumulate metrics
        results[loader_type]["acc"].append(float(acc))
        results[loader_type]["recall"].append(float(recall))
        results[loader_type]["f1"].append(float(f1))
        results[loader_type]["nacc"].append(float(nacc))
        results[loader_type]["oacc"].append(float(oacc))
    
    # Average and flatten
    flat_results = {}
    for stage, metrics in results.items():
        for metric, values in metrics.items():
            if values:  # Only include if we have values
                flat_results[f"{stage}/{metric}"] = np.nanmean(values)
    
    return flat_results
