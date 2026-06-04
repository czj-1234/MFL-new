# ============================================================
# MVSA-Multiple Data Processing for 4-Class Sentiment Classification
# Generate:
#   1. mvsa_4class_all.json
#   2. mvsa_4class_train.json
#   3. mvsa_4class_val.json
#   4. mvsa_4class_test.json
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

# 项目内相对路径
# 你的原始 MVSA 数据应该放在：
# data/MVSA/
DATA_ROOT = "data/MVSA"

LABEL_FILE = os.path.join(DATA_ROOT, "labelResultAll.txt")

# 处理后的 JSON 输出到这里
OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

ALL_JSON = os.path.join(OUT_DIR, "mvsa_4class_all.json")
TRAIN_JSON = os.path.join(OUT_DIR, "mvsa_4class_train.json")
VAL_JSON = os.path.join(OUT_DIR, "mvsa_4class_val.json")
TEST_JSON = os.path.join(OUT_DIR, "mvsa_4class_test.json")


# 4-class sentiment labels
label_map = {
    "negative": 0,
    "neutral_mixed": 1,
    "positive": 2,
    "strong_positive": 3,
}

id_to_label_name = {
    0: "negative",
    1: "neutral_mixed",
    2: "positive",
    3: "strong_positive",
}

valid_sentiments = {"positive", "neutral", "negative"}


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


def label_distribution(data):
    """
    Count label_name distribution.
    """
    return Counter([x["label_name"] for x in data])


def sentiment_strength_label_4class(votes):
    """
    Convert multiple sentiment votes into a 4-class sentiment label.

    Each MVSA-Multiple sample has up to 6 votes:
        annotator 1: text_label, image_label
        annotator 2: text_label, image_label
        annotator 3: text_label, image_label

    We compute:
        score = positive_count - negative_count

    First obtain the original sentiment-strength meaning:
        score <= -3        -> strong_negative
        score -2 or -1     -> weak_negative
        score == 0         -> neutral_mixed
        score 1 or 2       -> weak_positive
        score 3 or 4       -> medium_positive
        score 5 or 6       -> strong_positive

    Then merge into 4 classes:
        strong_negative + weak_negative  -> 0 negative
        neutral_mixed                    -> 1 neutral_mixed
        weak_positive + medium_positive  -> 2 positive
        strong_positive                  -> 3 strong_positive
    """
    valid_votes = [v for v in votes if v in valid_sentiments]

    if len(valid_votes) == 0:
        return None

    counter = Counter(valid_votes)

    positive_count = counter.get("positive", 0)
    neutral_count = counter.get("neutral", 0)
    negative_count = counter.get("negative", 0)

    score = positive_count - negative_count

    if score <= -1:
        class_id = 0
        class_name = "negative"
        original_label_name = "strong_negative" if score <= -3 else "weak_negative"
    elif score == 0:
        class_id = 1
        class_name = "neutral_mixed"
        original_label_name = "neutral_mixed"
    elif score in [1, 2, 3, 4]:
        class_id = 2
        class_name = "positive"
        original_label_name = "weak_positive" if score in [1, 2] else "medium_positive"
    elif score in [5, 6]:
        class_id = 3
        class_name = "strong_positive"
        original_label_name = "strong_positive"
    else:
        return None

    return {
        "label": class_id,
        "label_name": class_name,
        "original_label_name": original_label_name,
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

def generate_mvsa_4class_dataset(
    data_root=DATA_ROOT,
    label_file=LABEL_FILE,
    out_dir=OUT_DIR,
    seed=SEED,
):
    """
    Generate full MVSA 4-class JSON files.

    Outputs:
        mvsa_4class_all.json
        mvsa_4class_train.json
        mvsa_4class_val.json
        mvsa_4class_test.json
    """

    random.seed(seed)

    all_json = os.path.join(out_dir, "mvsa_4class_all.json")
    train_json = os.path.join(out_dir, "mvsa_4class_train.json")
    val_json = os.path.join(out_dir, "mvsa_4class_val.json")
    test_json = os.path.join(out_dir, "mvsa_4class_test.json")

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
    # Build 4-class sentiment labels
    # -----------------------------
    clean_label_items = []

    stats = {
        "raw_rows": len(df),
        "valid_4class_labels": 0,
        "invalid_or_empty_votes": 0,
        "missing_image": 0,
        "missing_text": 0,
        "empty_text": 0,
    }

    for _, row in df.iterrows():
        sample_id = str(row["ID"]).strip()

        votes = []

        # 后面三列是三个标注者结果，每个结果为 text_label,image_label
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

            if text_label in valid_sentiments:
                votes.append(text_label)

            if image_label in valid_sentiments:
                votes.append(image_label)

        label_info = sentiment_strength_label_4class(votes)

        if label_info is None:
            stats["invalid_or_empty_votes"] += 1
            continue

        stats["valid_4class_labels"] += 1

        clean_label_items.append({
            "id": sample_id,
            "label": label_info["label"],
            "label_name": label_info["label_name"],
            "original_label_name": label_info["original_label_name"],
            "positive_count": label_info["positive_count"],
            "neutral_count": label_info["neutral_count"],
            "negative_count": label_info["negative_count"],
            "score": label_info["score"],
            "num_votes": label_info["num_votes"],
            "raw_votes": label_info["raw_votes"],
        })

    print("\n4-class label construction stats:")
    for k, v in stats.items():
        print(f"{k}: {v}")

    print("\n4-class label distribution before matching files:")
    print(Counter([x["label_name"] for x in clean_label_items]))

    print("\nOriginal 6-class label distribution before matching files:")
    print(Counter([x["original_label_name"] for x in clean_label_items]))

    print("\nScore distribution before matching files:")
    print(Counter([x["score"] for x in clean_label_items]))

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
            "text": text,
            "label": item["label"],
            "label_name": item["label_name"],
            "original_label_name": item["original_label_name"],
            "positive_count": item["positive_count"],
            "neutral_count": item["neutral_count"],
            "negative_count": item["negative_count"],
            "score": item["score"],
            "num_votes": item["num_votes"],
            "raw_votes": item["raw_votes"],
        })

    print("\nAfter matching image and text:")
    print("Clean samples with image and text:", len(clean_samples))
    print("Missing image:", stats["missing_image"])
    print("Missing text:", stats["missing_text"])
    print("Empty text:", stats["empty_text"])

    print("\nFinal 4-class label distribution:")
    print(label_distribution(clean_samples))

    print("\nFinal original 6-class label distribution:")
    print(Counter([x["original_label_name"] for x in clean_samples]))

    print("\nFinal score distribution:")
    print(Counter([x["score"] for x in clean_samples]))

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
    labels = [x["label"] for x in clean_samples]

    label_counts = Counter(labels)
    print("\nLabel counts before split:")
    print(label_counts)

    min_count = min(label_counts.values())

    if min_count < 2:
        raise ValueError(
            f"At least one class has fewer than 2 samples: {label_counts}. "
            "Cannot do stratified train/val/test split."
        )

    # 80% train, 20% temp
    train_data, temp_data = train_test_split(
        clean_samples,
        test_size=0.2,
        random_state=seed,
        stratify=labels,
    )

    temp_labels = [x["label"] for x in temp_data]
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
    print("Train:", len(train_data), label_distribution(train_data))
    print("Val:  ", len(val_data), label_distribution(val_data))
    print("Test: ", len(test_data), label_distribution(test_data))

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
    generate_mvsa_4class_dataset()