# ============================================================
# Hateful Memes Data Processing for 2-Class Classification
#
# Use official seen split only:
#   train.jsonl     -> hateful_train.json
#   dev_seen.jsonl  -> hateful_val.json
#   test_seen.jsonl -> hateful_test.json
#
# Not used in this first version:
#   dev_unseen.jsonl
#   test_unseen.jsonl
#
# Output:
#   data/processed/hateful_all.json
#   data/processed/hateful_train.json
#   data/processed/hateful_val.json
#   data/processed/hateful_test.json
#
# For compatibility with existing FL pipeline:
#   label       = original Hateful Memes label
#   text_label  = label
#   image_label = label
#
# Label mapping:
#   0 = non_hateful
#   1 = hateful
# ============================================================

import os
import json
import random
from collections import Counter


# ============================================================
# 1. Configuration
# ============================================================

SEED = 42
random.seed(SEED)

# Expected structure:
# data/hateful_memes/
#   img/
#   train.jsonl
#   dev_seen.jsonl
#   dev_unseen.jsonl
#   test_seen.jsonl
#   test_unseen.jsonl
DATA_ROOT = "data/hateful_memes"

OUT_DIR = "data/processed"

TRAIN_FILE = os.path.join(DATA_ROOT, "train.jsonl")
VAL_FILE = os.path.join(DATA_ROOT, "dev_seen.jsonl")
TEST_FILE = os.path.join(DATA_ROOT, "test_seen.jsonl")

ALL_JSON = os.path.join(OUT_DIR, "hateful_all.json")
TRAIN_JSON = os.path.join(OUT_DIR, "hateful_train.json")
VAL_JSON = os.path.join(OUT_DIR, "hateful_val.json")
TEST_JSON = os.path.join(OUT_DIR, "hateful_test.json")

ID2LABEL = {
    0: "non_hateful",
    1: "hateful",
}


# ============================================================
# 2. Helper Functions
# ============================================================

def read_jsonl(path):
    """
    Read a JSONL file.
    """
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line == "":
                continue

            rows.append(json.loads(line))

    return rows


def save_json(data, path):
    """
    Save data as JSON.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def label_distribution(data, label_key="label_name"):
    """
    Count label distribution.
    """
    return Counter([x[label_key] for x in data])


def resolve_image_path(data_root, raw_img_path):
    """
    Resolve image path from raw Hateful Memes row.

    Raw examples usually contain:
        "img": "img/xxxxx.png"
    """
    if raw_img_path is None:
        return None

    raw_img_path = str(raw_img_path).strip()

    if raw_img_path == "":
        return None

    # Absolute path.
    if os.path.isabs(raw_img_path):
        return raw_img_path

    # Relative to DATA_ROOT.
    candidate = os.path.join(data_root, raw_img_path)

    if os.path.exists(candidate):
        return candidate

    # Fallback: search by basename under DATA_ROOT.
    basename = os.path.basename(raw_img_path)

    for dirpath, _, filenames in os.walk(data_root):
        if basename in filenames:
            return os.path.join(dirpath, basename)

    return candidate


def get_label(row):
    """
    Extract binary label from raw row.
    """
    if "label" not in row:
        return None

    label = row["label"]

    if isinstance(label, str):
        label = label.strip().lower()

        if label in [
            "0",
            "non_hateful",
            "non-hateful",
            "not_hateful",
            "not hateful",
            "benign",
        ]:
            return 0

        if label in [
            "1",
            "hateful",
            "hate",
        ]:
            return 1

        raise ValueError(f"Unknown label string: {label}")

    label = int(label)

    if label not in [0, 1]:
        raise ValueError(f"Invalid binary label: {label}")

    return label


def convert_row(row, split_name, data_root=DATA_ROOT):
    """
    Convert one raw Hateful Memes row into project-compatible format.
    """
    label = get_label(row)

    if label is None:
        return None

    label_name = ID2LABEL[label]

    sample_id = str(row.get("id", "")).strip()

    raw_img = (
        row.get("img")
        or row.get("image")
        or row.get("image_path")
        or row.get("path")
    )

    image_path = resolve_image_path(data_root, raw_img)

    text = str(row.get("text", "")).strip()

    if sample_id == "":
        sample_id = os.path.splitext(os.path.basename(str(raw_img)))[0]

    if text == "":
        print(f"[Warning] Empty text skipped: id={sample_id}")
        return None

    if image_path is None or not os.path.exists(image_path):
        print(f"[Warning] Missing image skipped: id={sample_id}, image={image_path}")
        return None

    return {
        "id": sample_id,
        "split": split_name,

        # image keys compatible with existing dataset class
        "image": image_path,
        "image_path": image_path,

        # text
        "text": text,

        # unified binary label
        "label": label,
        "label_name": label_name,

        # modality-specific labels copied from unified label
        "text_label": label,
        "text_label_name": label_name,
        "image_label": label,
        "image_label_name": label_name,

        # optional raw metadata
        "raw_img": raw_img,
    }


def load_split(jsonl_path, split_name, data_root=DATA_ROOT):
    """
    Load one official split.
    """
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"Cannot find {split_name} file: {jsonl_path}")

    print(f"\nLoading {split_name} from:")
    print(jsonl_path)

    rows = read_jsonl(jsonl_path)
    samples = []

    skipped_unlabeled = 0
    skipped_invalid = 0

    for row in rows:
        try:
            item = convert_row(
                row=row,
                split_name=split_name,
                data_root=data_root,
            )

            if item is None:
                if "label" not in row:
                    skipped_unlabeled += 1
                else:
                    skipped_invalid += 1
                continue

            samples.append(item)

        except Exception as e:
            skipped_invalid += 1
            print(f"[Warning] Failed to convert row in {split_name}: {e}")

    print(f"{split_name} raw rows:", len(rows))
    print(f"{split_name} clean samples:", len(samples))
    print(f"{split_name} skipped unlabeled:", skipped_unlabeled)
    print(f"{split_name} skipped invalid:", skipped_invalid)
    print(f"{split_name} label distribution:", label_distribution(samples))

    if len(samples) == 0:
        raise ValueError(
            f"No valid labeled samples found for split={split_name}. "
            f"Please check whether {jsonl_path} contains labels and valid image paths."
        )

    return samples


# ============================================================
# 3. Main Processing Function
# ============================================================

def generate_hateful_2class_dataset(
    data_root=DATA_ROOT,
    train_file=TRAIN_FILE,
    val_file=VAL_FILE,
    test_file=TEST_FILE,
    out_dir=OUT_DIR,
):
    """
    Generate Hateful Memes 2-class JSON files using seen splits only.
    """

    if not os.path.exists(data_root):
        raise FileNotFoundError(f"DATA_ROOT does not exist: {data_root}")

    os.makedirs(out_dir, exist_ok=True)

    print("DATA_ROOT:", data_root)
    print("OUT_DIR:", out_dir)
    print("Using splits:")
    print("  train:", train_file)
    print("  val  :", val_file)
    print("  test :", test_file)

    train_data = load_split(
        jsonl_path=train_file,
        split_name="train",
        data_root=data_root,
    )

    val_data = load_split(
        jsonl_path=val_file,
        split_name="val_seen",
        data_root=data_root,
    )

    test_data = load_split(
        jsonl_path=test_file,
        split_name="test_seen",
        data_root=data_root,
    )

    all_data = train_data + val_data + test_data

    all_json = os.path.join(out_dir, "hateful_all.json")
    train_json = os.path.join(out_dir, "hateful_train.json")
    val_json = os.path.join(out_dir, "hateful_val.json")
    test_json = os.path.join(out_dir, "hateful_test.json")

    save_json(all_data, all_json)
    save_json(train_data, train_json)
    save_json(val_data, val_json)
    save_json(test_data, test_json)

    print("\nFinal split summary:")

    print("Train:", len(train_data))
    print("  label:", label_distribution(train_data))
    print("  text :", Counter([x["text_label_name"] for x in train_data]))
    print("  image:", Counter([x["image_label_name"] for x in train_data]))

    print("Val:  ", len(val_data))
    print("  label:", label_distribution(val_data))
    print("  text :", Counter([x["text_label_name"] for x in val_data]))
    print("  image:", Counter([x["image_label_name"] for x in val_data]))

    print("Test: ", len(test_data))
    print("  label:", label_distribution(test_data))
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
    generate_hateful_2class_dataset()