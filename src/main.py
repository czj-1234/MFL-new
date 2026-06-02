# ============================================================
# Main Entry: Run MVSA Structural Baseline Experiment
# ============================================================

import os
import torch
import pandas as pd

from src.federated import run_experiment
from src.utils import set_seed


# ============================================================
# 1. Args
# ============================================================

class Args:
    # data
    train_json = "data/processed/mvsa_6class_train.json"
    val_json = "data/processed/mvsa_6class_val.json"
    test_json = "data/processed/mvsa_6class_test.json"

    # tokenizer and text model
    tokenizer_name = "distilbert-base-uncased"
    text_model_name = "distilbert-base-uncased"

    # label setting
    num_classes = 6

    # structure
    # Choose one:
    # "image_only"
    # "text_only"
    # "modality_exclusive"
    setting_name = "modality_exclusive"

    # Choose one:
    # "iid"
    # "0.3"
    # "0.7"
    # "1.0"
    association = "iid"

    # federated setting
    num_clients = 6

    # None = 自动尽量使用完整 train 数据
    # 如果你只是想快速测试，可以先改成 100
    samples_per_client = 20

    allow_overlap = False

    # training
    rounds = 1
    local_epochs = 1
    lr = 5e-4
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # local training
    batch_size = 32
    max_local_steps = None

    # evaluation
    eval_batch_size = 64
    max_train_eval_samples = 300
    max_val_eval_samples = None
    max_test_eval_samples = None

    # loading
    max_text_len = 64
    num_workers = 0

    # model dimensions
    image_hidden_dim = 128
    text_hidden_dim = 128
    projector_hidden_dim = 128
    dropout = 0.1

    # pretrained encoder control
    freeze_image_backbone = True
    freeze_text_backbone = True
    pretrained_image = True

    # update extraction target
    target_patterns = [
        "multi_modal_projector.0.weight",
        "multi_modal_projector.0.bias",
        "classifier.weight",
        "classifier.bias",
    ]

    # output
    output_root = "results/structure_baseline"
    modality_dir = os.path.join(output_root, setting_name)
    out_dir = os.path.join(modality_dir, association)

    # analysis rounds
    analysis_rounds = {1, 5, 10, 20, 30, 50, 80, 100}


# ============================================================
# 2. Run one experiment
# ============================================================

def run_one():
    args = Args()
    set_seed(args.seed)

    print("Running MVSA Clean FL Structural Baseline")
    print("Model: ResNet18 + DistilBERT")
    print("SETTING_NAME:", args.setting_name)
    print("ASSOCIATION:", args.association)
    print("DEVICE:", args.device)
    print("TRAIN_JSON:", args.train_json)
    print("VAL_JSON:", args.val_json)
    print("TEST_JSON:", args.test_json)
    print("OUT_DIR:", args.out_dir)

    summary, round_logs, all_mat = run_experiment(args)

    print("\nDone.")
    print("Summary:")
    for k, v in summary.items():
        print(f"{k}: {v}")


# ============================================================
# 3. Run all experiments
# ============================================================

def run_all():
    all_summaries = []

    for setting_name in ["image_only", "text_only", "modality_exclusive"]:
        for association in ["iid", "0.3", "0.7", "1.0"]:

            print("\n" + "=" * 80)
            print(f"Running setting={setting_name}, association={association}")
            print("=" * 80)

            class RunArgs(Args):
                pass

            args = RunArgs()
            args.setting_name = setting_name
            args.association = association

            args.modality_dir = os.path.join(args.output_root, setting_name)
            args.out_dir = os.path.join(args.modality_dir, association)

            set_seed(args.seed)

            summary, round_logs, all_mat = run_experiment(args)
            all_summaries.append(summary)

    summary_df = pd.DataFrame(all_summaries)

    preferred_columns = [
        "setting_name",
        "association",
        "num_clients",
        "samples_per_client",
        "allow_overlap",
        "rounds",
        "local_epochs",
        "lr",
        "seed",
        "model",
        "freeze_image_backbone",
        "freeze_text_backbone",
        "global_acc",
        "test_acc",
        "e_rank",
        "Top1_ratio",
        "Top3_ratio",
        "Top5_ratio",
        "Silhouette",
        "DBI",
        "CHI",
        "kmeans_acc",
        "attack_success_rate",
    ]

    summary_df = summary_df[
        [c for c in preferred_columns if c in summary_df.columns]
    ]

    summary_csv = os.path.join(
        Args.output_root,
        "all_structure_baseline_summary.csv",
    )

    os.makedirs(os.path.dirname(summary_csv), exist_ok=True)
    summary_df.to_csv(summary_csv, index=False)

    print("\nAll experiments done.")
    print("Saved summary CSV to:", summary_csv)
    print(summary_df)


# ============================================================
# 4. Entry
# ============================================================

if __name__ == "__main__":
    # 先跑一个实验确认没问题
    run_one()

    # 如果你想一次跑 12 个实验，把上面 run_one() 注释掉，
    # 然后取消下面这一行注释：
    # run_all()