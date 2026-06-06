# ============================================================
# MVSA-Multiple Data Processing for Original 3-Class Sentiment Classification
#
# Generate:
#   1. mvsa_3class_all.json
#   2. mvsa_3class_train.json
#   3. mvsa_3class_val.json
#   4. mvsa_3class_test.json
#
# Important:
#   - This version DOES NOT merge text labels and image labels.
#   - Each sample keeps:
#       text_label
#       text_label_name
#       image_label
#       image_label_name
#
# For heterogeneous modality-exclusive FL:
#   - text clients use text_label
#   - image clients use image_label
# ============================================================

import os
import json
import random
from collections import Counter

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# 1. Configuration
# ============================================================

SEED = 42
random.seed(SEED)

# Your original MVSA-Multiple data should be placed under:
# data/MVSA/
DATA_ROOT = "data/MVSA"

LABEL_FILE = os.path.join(DATA_ROOT, "labelResultAll.txt")

# Processed JSON output directory
OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

ALL_JSON = os.path.join(OUT_DIR, "mvsa_3class_all.json")
TRAIN_JSON = os.path.join(OUT_DIR, "mvsa_3class_train.json")
VAL_JSON = os.path.join(OUT_DIR, "mvsa_3class_val.json")
TEST_JSON = os.path.join(OUT_DIR, "mvsa_3class_test.json")


# 3-class sentiment labels
LABEL2ID = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

ID2LABEL = {
    0: "negative",
    1: "neutral",
    2: "positive",
}

VALID_SENTIMENTS = {"positive", "neutral", "negative"}


# ============================================================
# 2. Helper Functions
# ============================================================

def build_file_index(root, exts):
    """
    Build an index from filename stem to full path.

    Example:
        2499.jpg -> index["2499"] = ".../2499.jpg"
        2499.txt -> index["2499"] = ".../2499.txt"
    """
    file_index = {}

    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            name, ext = os.path.splitext(filename)
            ext = ext.lower()

            if ext in exts:
                file_index[name] = os.path.join(dirpath, filename)

    return file_index


def read_text_file(path):
    """
    Read text file safely using several possible encodings.
    """
    if path is None:
        return ""

    for encoding in ["utf-8", "utf-8-sig", "latin-1", "gbk"]:
        try:
            with open(path, "r", encoding=encoding) as f:
                return f.read().strip()
        except Exception:
            continue

    return ""


def save_json(data, path):
    """
    Save data as JSON.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def label_distribution(data, label_key="label_name"):
    """
    Count label distribution by a given label name key.
    """
    return Counter([x[label_key] for x in data])


def resolve_3class_label(votes):
    """
    Resolve multiple annotations of ONE modality into a 3-class label.

    This function is used separately for:
        - text votes
        - image votes

    It does NOT mix text votes and image votes.

    Rule:
        score = positive_count - negative_count

        score > 0  -> positive
        score < 0  -> negative
        score == 0 -> neutral

    Why this rule:
        - It keeps the original 3-class task.
        - It avoids creating 4-class or 6-class labels.
        - It handles cases where votes are mixed.
    """
    valid_votes = [v for v in votes if v in VALID_SENTIMENTS]

    if len(valid_votes) == 0:
        return None

    counter = Counter(valid_votes)

    positive_count = counter.get("positive", 0)
    neutral_count = counter.get("neutral", 0)
    negative_count = counter.get("negative", 0)

    score = positive_count - negative_count

    if score > 0:
        label_name = "positive"
    elif score < 0:
        label_name = "negative"
    else:
        label_name = "neutral"

    label_id = LABEL2ID[label_name]

    return {
        "label": label_id,
        "label_name": label_name,
        "positive_count": positive_count,
        "neutral_count": neutral_count,
        "negative_count": negative_count,
        "score": score,
        "num_votes": len(valid_votes),
        "raw_votes": valid_votes,
    }


def read_label_file(label_file):
    """
    Try to read labelResultAll.txt robustly.

    The original file is usually tab-separated:
        ID    text,image    text,image    text,image

    Example row:
        1     positive,positive    neutral,positive    positive,neutral
    """
    try:
        df = pd.read_csv(label_file, sep="\t")
    except Exception:
        df = pd.read_csv(label_file, sep=None, engine="python")

    df.columns = [str(c).strip() for c in df.columns]

    return df


# ============================================================
# 3. Main Processing Function
# ============================================================

def generate_mvsa_3class_dataset(
    data_root=DATA_ROOT,
    label_file=LABEL_FILE,
    out_dir=OUT_DIR,
    seed=SEED,
):
    """
    Generate full MVSA original 3-class JSON files.

    Outputs:
        mvsa_3class_all.json
        mvsa_3class_train.json
        mvsa_3class_val.json
        mvsa_3class_test.json

    Each sample contains:
        id
        image
        text
        text_label
        text_label_name
        image_label
        image_label_name

    For training:
        text client  -> use text_label
        image client -> use image_label
    """

    random.seed(seed)

    all_json = os.path.join(out_dir, "mvsa_3class_all.json")
    train_json = os.path.join(out_dir, "mvsa_3class_train.json")
    val_json = os.path.join(out_dir, "mvsa_3class_val.json")
    test_json = os.path.join(out_dir, "mvsa_3class_test.json")

    os.makedirs(out_dir, exist_ok=True)

    # -----------------------------
    # Check paths
    # -----------------------------
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"DATA_ROOT does not exist: {data_root}")

    if not os.path.exists(label_file):
        raise FileNotFoundError(f"Cannot find labelResultAll.txt: {label_file}")

    print("DATA_ROOT:", data_root)
    print("LABEL_FILE:", label_file)
    print("OUT_DIR:", out_dir)

    # -----------------------------
    # Build image and text index
    # -----------------------------
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    text_exts = {".txt"}

    print("\nBuilding image index...")
    image_index = build_file_index(data_root, image_exts)
    print("Number of image files found:", len(image_index))

    print("\nBuilding text index...")
    text_index = build_file_index(data_root, text_exts)
    print("Number of text files found:", len(text_index))

    # -----------------------------
    # Read annotation file
    # -----------------------------
    df = read_label_file(label_file)

    print("\nRaw annotation file:")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print(df.head())

    if "ID" not in df.columns:
        raise ValueError(
            "Cannot find column 'ID' in labelResultAll.txt. "
            "Please check the file format."
        )

    # -----------------------------
    # Build separate text/image 3-class labels
    # -----------------------------
    clean_label_items = []

    stats = {
        "raw_rows": len(df),
        "valid_3class_labels": 0,
        "invalid_or_empty_text_votes": 0,
        "invalid_or_empty_image_votes": 0,
        "missing_image": 0,
        "missing_text": 0,
        "empty_text": 0,
    }

    for _, row in df.iterrows():
        sample_id = str(row["ID"]).strip()

        text_votes = []
        image_votes = []

        # After ID, each column is usually one annotator result:
        # text_label,image_label
        for col in df.columns[1:]:
            value = str(row[col]).strip()

            if value.lower() in ["nan", "none", ""]:
                continue

            if "," not in value:
                continue

            parts = value.split(",")

            if len(parts) < 2:
                continue

            text_label = parts[0].strip().lower()
            image_label = parts[1].strip().lower()

            if text_label in VALID_SENTIMENTS:
                text_votes.append(text_label)

            if image_label in VALID_SENTIMENTS:
                image_votes.append(image_label)

        text_label_info = resolve_3class_label(text_votes)
        image_label_info = resolve_3class_label(image_votes)

        if text_label_info is None:
            stats["invalid_or_empty_text_votes"] += 1
            continue

        if image_label_info is None:
            stats["invalid_or_empty_image_votes"] += 1
            continue

        stats["valid_3class_labels"] += 1

        clean_label_items.append({
            "id": sample_id,

            "text_label": text_label_info["label"],
            "text_label_name": text_label_info["label_name"],
            "text_positive_count": text_label_info["positive_count"],
            "text_neutral_count": text_label_info["neutral_count"],
            "text_negative_count": text_label_info["negative_count"],
            "text_score": text_label_info["score"],
            "text_num_votes": text_label_info["num_votes"],
            "text_raw_votes": text_label_info["raw_votes"],

            "image_label": image_label_info["label"],
            "image_label_name": image_label_info["label_name"],
            "image_positive_count": image_label_info["positive_count"],
            "image_neutral_count": image_label_info["neutral_count"],
            "image_negative_count": image_label_info["negative_count"],
            "image_score": image_label_info["score"],
            "image_num_votes": image_label_info["num_votes"],
            "image_raw_votes": image_label_info["raw_votes"],
        })

    print("\n3-class label construction stats:")
    for k, v in stats.items():
        print(f"{k}: {v}")

    print("\nText label distribution before matching files:")
    print(Counter([x["text_label_name"] for x in clean_label_items]))

    print("\nImage label distribution before matching files:")
    print(Counter([x["image_label_name"] for x in clean_label_items]))

    print("\nText score distribution before matching files:")
    print(Counter([x["text_score"] for x in clean_label_items]))

    print("\nImage score distribution before matching files:")
    print(Counter([x["image_score"] for x in clean_label_items]))

    # -----------------------------
    # Attach image path and text content
    # -----------------------------
    clean_samples = []

    for item in clean_label_items:
        sample_id = item["id"]

        image_path = image_index.get(sample_id)
        text_path = text_index.get(sample_id)

        if image_path is None:
            stats["missing_image"] += 1
            continue

        if text_path is None:
            stats["missing_text"] += 1
            continue

        text = read_text_file(text_path)

        if text == "":
            stats["empty_text"] += 1
            continue

        clean_samples.append({
            "id": sample_id,
            "image": image_path,
            "image_path": image_path,
            "text": text,

            "text_label": item["text_label"],
            "text_label_name": item["text_label_name"],
            "text_positive_count": item["text_positive_count"],
            "text_neutral_count": item["text_neutral_count"],
            "text_negative_count": item["text_negative_count"],
            "text_score": item["text_score"],
            "text_num_votes": item["text_num_votes"],
            "text_raw_votes": item["text_raw_votes"],

            "image_label": item["image_label"],
            "image_label_name": item["image_label_name"],
            "image_positive_count": item["image_positive_count"],
            "image_neutral_count": item["image_neutral_count"],
            "image_negative_count": item["image_negative_count"],
            "image_score": item["image_score"],
            "image_num_votes": item["image_num_votes"],
            "image_raw_votes": item["image_raw_votes"],
        })

    print("\nAfter matching image and text:")
    print("Clean samples with image and text:", len(clean_samples))
    print("Missing image:", stats["missing_image"])
    print("Missing text:", stats["missing_text"])
    print("Empty text:", stats["empty_text"])

    print("\nFinal text 3-class label distribution:")
    print(Counter([x["text_label_name"] for x in clean_samples]))

    print("\nFinal image 3-class label distribution:")
    print(Counter([x["image_label_name"] for x in clean_samples]))

    print("\nFinal text score distribution:")
    print(Counter([x["text_score"] for x in clean_samples]))

    print("\nFinal image score distribution:")
    print(Counter([x["image_score"] for x in clean_samples]))

    if len(clean_samples) == 0:
        raise ValueError("No clean samples found. Please check DATA_ROOT and file structure.")

    # -----------------------------
    # Save all clean data
    # -----------------------------
    save_json(clean_samples, all_json)

    print("\nSaved all clean samples:")
    print(all_json)

    # -----------------------------
    # Train / Val / Test split
    # -----------------------------
    # Since modality_exclusive uses both text_label and image_label,
    # we create a stable split based on text_label for stratification.
    #
    # This does NOT mean image clients use text_label.
    # It is only for splitting samples into train/val/test.
    #
    # During training:
    #   text client  -> text_label
    #   image client -> image_label
    split_labels = [x["text_label"] for x in clean_samples]

    split_label_counts = Counter(split_labels)
    print("\nText label counts before split:")
    print(split_label_counts)

    min_count = min(split_label_counts.values())

    if min_count < 2:
        raise ValueError(
            f"At least one class has fewer than 2 samples: {split_label_counts}. "
            "Cannot do stratified train/val/test split."
        )

    # 80% train, 20% temp
    train_data, temp_data = train_test_split(
        clean_samples,
        test_size=0.2,
        random_state=seed,
        stratify=split_labels,
    )

    temp_labels = [x["text_label"] for x in temp_data]
    temp_counts = Counter(temp_labels)

    if min(temp_counts.values()) < 2:
        print("\nWarning: Some classes in temp split have fewer than 2 samples.")
        print("Using non-stratified val/test split for temp_data.")

        val_data, test_data = train_test_split(
            temp_data,
            test_size=0.5,
            random_state=seed,
            stratify=None,
        )
    else:
        # 10% val, 10% test
        val_data, test_data = train_test_split(
            temp_data,
            test_size=0.5,
            random_state=seed,
            stratify=temp_labels,
        )

    save_json(train_data, train_json)
    save_json(val_data, val_json)
    save_json(test_data, test_json)

    # -----------------------------
    # Final summary
    # -----------------------------
    print("\nFinal split summary:")
    print("Train:", len(train_data))
    print("  text :", Counter([x["text_label_name"] for x in train_data]))
    print("  image:", Counter([x["image_label_name"] for x in train_data]))

    print("Val:  ", len(val_data))
    print("  text :", Counter([x["text_label_name"] for x in val_data]))
    print("  image:", Counter([x["image_label_name"] for x in val_data]))

    print("Test: ", len(test_data))
    print("  text :", Counter([x["text_label_name"] for x in test_data]))
    print("  image:", Counter([x["image_label_name"] for x in test_data]))

    print("\nSaved JSON files:")
    print("ALL_JSON  =", all_json)
    print("TRAIN_JSON=", train_json)
    print("VAL_JSON  =", val_json)
    print("TEST_JSON =", test_json)

    print("\nDone.")

    return {
        "all_json": all_json,
        "train_json": train_json,
        "val_json": val_json,
        "test_json": test_json,
    }


# ============================================================
# 4. Run directly
# ============================================================

if __name__ == "__main__":
    generate_mvsa_3class_dataset()