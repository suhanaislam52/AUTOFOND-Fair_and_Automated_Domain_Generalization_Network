import argparse
import logging

import wandb
import yaml
from dotenv import load_dotenv

from src.train.fit import fit
from src.utils.hparams import random_hparams, seed_hash
from src.utils.misc import config_logging

NUM_TEST_SETS = 4


def main():
    run = wandb.init()

    # Access experiment parameters
    overall_seed = wandb.config.overall_seed
    data_dir = wandb.config.data_dir
    log_dir = wandb.config.log_dir
    num_workers = wandb.config.num_workers
    holdout_fraction = wandb.config.holdout_fraction
    n_steps = wandb.config.n_steps
    checkpoint_freq = wandb.config.checkpoint_freq
    model_checkpoint = wandb.config.model_checkpoint
    teacher_paths = (
        wandb.config.teacher_paths if "teacher_paths" in wandb.config else None
    )

    # Access sweep configuration
    algo = wandb.config.algo
    dataset = wandb.config.dataset
    test_set_id = wandb.config.test_set_id
    hparam_id = wandb.config.hparam_id
    trial_id = wandb.config.trial_id
    overlap = wandb.config.overlap

    fit(
        id="{}-{}-{}".format(run.project, run.sweep_id, run.id),
        seed=overall_seed,
        trial_seed=trial_id,
        hparams_seed=hparam_id,
        algorithm_name=algo,
        dataset_name=dataset,
        data_dir=data_dir,
        log_dir=log_dir,
        num_workers=num_workers,
        test_envs=[test_set_id],
        overlap_type=overlap,
        holdout_fraction=holdout_fraction,
        n_steps=n_steps,
        checkpoint_freq=checkpoint_freq,
        model_checkpoint=model_checkpoint,
        teacher_paths=teacher_paths,
        num_domain_linked_classes=wandb.config.num_domain_linked_classes,
        num_classes=wandb.config.num_classes
    )

    wandb.finish()


if __name__ == "__main__":
    # load enviroment variables
    load_dotenv()

    # load cmd line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_entity", type=str, required=True)
    parser.add_argument("--wandb_project", type=str, required=True)
    parser.add_argument("--config_file", type=str, default="config/sweep_config.yaml")
    parser.add_argument("--sweep_id", type=str, default=None)
    
    # format logger
    config_logging()

    args = parser.parse_args()
    sweep_id = args.sweep_id
    config_file = args.config_file
    wandb_entity = args.wandb_entity
    wandb_project = args.wandb_project

    if sweep_id is None:
        with open(config_file, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        sweep_id = wandb.sweep(
            sweep=config,
            entity=wandb_entity,
            project=wandb_project,
        )

    wandb.agent(
        sweep_id, 
        function=main,
        entity=wandb_entity,
        project=wandb_project
    )

    

    

    

