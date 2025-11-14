"""Unit tests."""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
import unittest
import uuid
from typing import List, Tuple

import torch
from domainbed import algorithms, hparams_registry, networks
from domainbed.test import helpers
from parameterized import parameterized

from src.datasets import PACS, VLCS


def get_overlap_params() -> List[Tuple[str, str, int, List[int]]]:
    datasets = [PACS, VLCS]

    params = []
    for dataset in datasets:
        for overlap_type in dataset.OVERLAP_CONFIG:
            for test_env in range(len(dataset.ENVIRONMENTS)):
                id = f"{dataset.__name__}_{overlap_type}_test_{test_env}"
                params.append((id, dataset, overlap_type, test_env))

    return params


class TestOverlapDatasets(unittest.TestCase):
    @parameterized.expand(get_overlap_params())
    @unittest.skipIf(
        "DATA_DIR" not in os.environ, "needs DATA_DIR environment " "variable"
    )
    def test_overlap_datasets(self, _, dataset, overlap, test_env):
        """
        Test that class filters remove classes from enviroment datasets
        """
        hparams = hparams_registry.default_hparams("ERM", dataset.__name__)
        dataset = dataset(
            os.environ["DATA_DIR"], [test_env], hparams, overlap, overlap_seed=0
        )
        self.assertEqual(len(dataset.ENVIRONMENTS), len(dataset))

        for env in dataset:
            targets = set(env.targets)
            # print(
            #     f"{dataset_name} {class_overlap_id}%, ",
            #     f"env:{env.env_name}, test:{env.is_test_env}, ",
            #     f"allowed:{env.allowed_classes}, targets:{targets}, ",
            #     f"remove_classes:{env.remove_classes}",
            # )
            self.assertEqual(
                set(env.allowed_classes),
                targets,
                (
                    f"{dataset.__name__} {overlap}%, "
                    f"env:{env.env_name}, test:{env.is_test_env}, "
                    f"allowed:{env.allowed_classes}, targets:{targets}, "
                    f"remove_classes:{env.remove_classes}"
                ),
            )
            if env.is_test_env:
                self.assertEqual(len(targets), dataset.num_classes)
