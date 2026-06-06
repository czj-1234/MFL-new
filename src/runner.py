# ============================================================
# Experiment Runner
# ============================================================

import os

import pandas as pd

from src.config import ExperimentArgs
from src.federated import run_experiment
from src.utils import set_seed


def run_one_experiment(args):
    """
    Run one experiment.
    """
    set_seed(args.seed)

    print("Running MVSA Original 3-Class FL Structural Baseline")
    print("Model: CLIP-ViT-B/32 + RoBERTa-base")
    print("TASK:", "original 3-class modality-specific sentiment classification")
    print("NUM_CLASSES:", args.num_classes)
    print("NUM_CLIENTS:", args.num_clients)
    print("SETTING_NAME:", args.setting_name)
    print("ASSOCIATION:", args.association)
    print("PARTITION_MODE:", getattr(args, "partition_mode", "fixed"))
    print("DEVICE:", args.device)
    print("TRAIN_JSON:", args.train_json)
    print("VAL_JSON:", args.val_json)
    print("TEST_JSON:", args.test_json)
    print("OUT_DIR:", args.out_dir)

    if args.setting_name == "text_only":
        print("CLIENT_MODALITIES: all clients=text")
        print("LABEL_SOURCE: text_label")

    elif args.setting_name == "image_only":
        print("CLIENT_MODALITIES: all clients=image")
        print("LABEL_SOURCE: image_label")

    elif args.setting_name == "modality_exclusive":
        print("CLIENT_MODALITIES: even clients=image, odd clients=text")
        print("CLIENT DESIGN:")
        print("  client 0: image, negative-dominant")
        print("  client 1: text,  neutral-dominant")
        print("  client 2: image, positive-dominant")
        print("  client 3: text,  negative-dominant")
        print("  client 4: image, neutral-dominant")
        print("  client 5: text,  positive-dominant")
        print("LABEL_SOURCE: text clients use text_label; image clients use image_label")

    elif args.setting_name == "full_multimodal":
        print("CLIENT_MODALITIES: all clients=both")
        print("LABEL_SOURCE: auto / fixed label if available")

    else:
        print("LABEL_SOURCE: unknown")

    summary, round_logs, all_mat = run_experiment(args)

    print("\nDone.")
    print("Summary:")
    for k, v in summary.items():
        print(f"{k}: {v}")

    return summary, round_logs, all_mat


def run_all_experiments(cfg, cli_args):
    """
    Run all structural baseline experiments.
    """
    all_summaries = []

    setting_list = [
        "image_only",
        "text_only",
        "modality_exclusive",
    ]

    association_list = [
        "iid",
        "0.3",
        "0.7",
        "1.0",
    ]

    for setting_name in setting_list:
        for association in association_list:
            print("\n" + "=" * 80)
            print(f"Running 3-class setting={setting_name}, association={association}")
            print("=" * 80)

            args = ExperimentArgs(
                cfg,
                setting_name=setting_name,
                association=association,
                rounds=cli_args.rounds,
                samples_per_client=cli_args.samples_per_client,
                output_root=cli_args.output_root,
            )

            summary, round_logs, all_mat = run_one_experiment(args)
            all_summaries.append(summary)

    summary_df = pd.DataFrame(all_summaries)

    preferred_columns = [
        "setting_name",
        "association",
        "num_clients",
        "samples_per_client",
        "partition_mode",
        "allow_overlap",
        "rounds",
        "local_epochs",
        "lr",
        "weight_decay",
        "seed",
        "model",
        "freeze_image_backbone",
        "freeze_text_backbone",

        # task definition
        "task_type",
        "global_num_classes",
        "random_chance_acc",
        "label_space_split_by_modality",
        "uses_modality_specific_labels",
        "text_client_label_source",
        "image_client_label_source",

        # train utility metrics
        "train_acc",
        "train_macro_f1",
        "train_macro_precision",
        "train_macro_recall",
        "train_balanced_acc",

        # train per-class F1
        "train_f1_negative",
        "train_f1_neutral",
        "train_f1_positive",

        # validation/global utility metrics
        "global_acc",
        "global_macro_f1",
        "global_macro_precision",
        "global_macro_recall",
        "global_balanced_acc",

        # validation per-class F1
        "val_f1_negative",
        "val_f1_neutral",
        "val_f1_positive",

        # test utility metrics
        "test_acc",
        "test_macro_f1",
        "test_macro_precision",
        "test_macro_recall",
        "test_balanced_acc",

        # test per-class F1
        "test_f1_negative",
        "test_f1_neutral",
        "test_f1_positive",

        # modality-specific metrics, mainly for modality_exclusive
        "train_text_acc",
        "train_image_acc",
        "train_text_macro_f1",
        "train_image_macro_f1",
        "val_text_acc",
        "val_image_acc",
        "val_text_macro_f1",
        "val_image_macro_f1",
        "test_text_acc",
        "test_image_acc",
        "test_text_macro_f1",
        "test_image_macro_f1",

        # modality-specific per-class F1
        "train_text_f1_negative",
        "train_text_f1_neutral",
        "train_text_f1_positive",
        "train_image_f1_negative",
        "train_image_f1_neutral",
        "train_image_f1_positive",

        "val_text_f1_negative",
        "val_text_f1_neutral",
        "val_text_f1_positive",
        "val_image_f1_negative",
        "val_image_f1_neutral",
        "val_image_f1_positive",

        "test_text_f1_negative",
        "test_text_f1_neutral",
        "test_text_f1_positive",
        "test_image_f1_negative",
        "test_image_f1_neutral",
        "test_image_f1_positive",

        # accuracy above random chance
        "train_acc_above_chance",
        "val_acc_above_chance",
        "global_acc_above_chance",
        "test_acc_above_chance",

        # structure metrics
        "e_rank",
        "Top1_ratio",
        "Top3_ratio",
        "Top5_ratio",
        "Silhouette",
        "DBI",
        "CHI",
        "kmeans_acc",

        # attack metrics
        "attack_success_rate_rf",
        "attack_success_rate_mlp",
        "attack_success_rate_xgb",
        "attack_success_rate_mean",
    ]

    summary_df = summary_df[
        [c for c in preferred_columns if c in summary_df.columns]
    ]

    output_root = cli_args.output_root or cfg["experiment"]["output_root"]

    summary_csv = os.path.join(
        output_root,
        "all_structure_baseline_summary.csv",
    )

    os.makedirs(os.path.dirname(summary_csv), exist_ok=True)
    summary_df.to_csv(summary_csv, index=False)

    print("\nAll 3-class experiments done.")
    print("Saved summary CSV to:", summary_csv)
    print(summary_df)

    return summary_df