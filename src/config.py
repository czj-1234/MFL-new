# ============================================================
# Config and CLI Arguments
# ============================================================

import argparse
import os

import torch

from src.utils import load_config


class ExperimentArgs:
    """
    Convert config dict into an args-like object used by run_experiment().
    """

    def __init__(
        self,
        cfg,
        setting_name=None,
        association=None,
        rounds=None,
        samples_per_client=None,
        output_root=None,
    ):
        # seed
        self.seed = cfg["seed"]

        # data
        self.train_json = cfg["data"]["train_json"]
        self.val_json = cfg["data"]["val_json"]
        self.test_json = cfg["data"]["test_json"]
        self.num_classes = cfg["data"]["num_classes"]

        # model
        self.tokenizer_name = cfg["model"]["tokenizer_name"]
        self.text_model_name = cfg["model"]["text_model_name"]
        self.image_hidden_dim = cfg["model"]["image_hidden_dim"]
        self.text_hidden_dim = cfg["model"]["text_hidden_dim"]
        self.projector_hidden_dim = cfg["model"]["projector_hidden_dim"]
        self.dropout = cfg["model"]["dropout"]
        self.freeze_image_backbone = cfg["model"]["freeze_image_backbone"]
        self.freeze_text_backbone = cfg["model"]["freeze_text_backbone"]
        self.pretrained_image = cfg["model"]["pretrained_image"]

        # federated
        self.num_clients = cfg["federated"]["num_clients"]
        self.partition_mode = cfg["federated"].get("partition_mode", "fixed")
        self.samples_per_client = cfg["federated"]["samples_per_client"]
        self.allow_overlap = cfg["federated"]["allow_overlap"]
        self.rounds = cfg["federated"]["rounds"]
        self.local_epochs = cfg["federated"]["local_epochs"]
        self.lr = cfg["federated"]["lr"]
        self.batch_size = cfg["federated"]["batch_size"]
        self.max_local_steps = cfg["federated"]["max_local_steps"]

        # evaluation
        self.eval_batch_size = cfg["evaluation"]["eval_batch_size"]
        self.max_train_eval_samples = cfg["evaluation"]["max_train_eval_samples"]
        self.max_val_eval_samples = cfg["evaluation"]["max_val_eval_samples"]
        self.max_test_eval_samples = cfg["evaluation"]["max_test_eval_samples"]
        self.max_text_len = cfg["evaluation"]["max_text_len"]
        self.num_workers = cfg["evaluation"]["num_workers"]

        # experiment
        self.setting_name = cfg["experiment"]["setting_name"]
        self.association = cfg["experiment"]["association"]
        self.output_root = cfg["experiment"]["output_root"]
        self.analysis_rounds = set(cfg["experiment"]["analysis_rounds"])

        # update extraction
        self.target_patterns = cfg["update"]["target_patterns"]

        # command-line overrides
        if setting_name is not None:
            self.setting_name = setting_name

        if association is not None:
            self.association = association

        if rounds is not None:
            self.rounds = rounds
            self.analysis_rounds = {1, self.rounds}

        # If samples_per_client is manually provided from CLI,
        # switch back to fixed-size client sampling.
        if samples_per_client is not None:
            self.samples_per_client = samples_per_client
            self.partition_mode = "fixed"

        if output_root is not None:
            self.output_root = output_root

        # device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # output path
        self.modality_dir = os.path.join(self.output_root, self.setting_name)
        self.out_dir = os.path.join(self.modality_dir, self.association)


def parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Run MVSA multimodal federated learning experiments."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config YAML file.",
    )

    parser.add_argument(
        "--setting",
        type=str,
        default=None,
        choices=["image_only", "text_only", "modality_exclusive", "full_multimodal"],
        help="Structural setting.",
    )

    parser.add_argument(
        "--association",
        type=str,
        default=None,
        choices=["iid", "0.3", "0.7", "1.0"],
        help="Association setting.",
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Override number of FL rounds.",
    )

    parser.add_argument(
        "--samples_per_client",
        type=int,
        default=None,
        help="Override samples per client. If provided, partition_mode becomes fixed.",
    )

    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Override output root.",
    )

    parser.add_argument(
        "--run_all",
        action="store_true",
        help="Run all experiments.",
    )

    return parser.parse_args()


def load_cli_config(cli_args):
    """
    Load config file from CLI args.
    """
    return load_config(cli_args.config)