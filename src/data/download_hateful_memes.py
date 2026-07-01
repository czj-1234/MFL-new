# ============================================================
# Download Hateful Memes dataset from KaggleHub
# Save to: data/hateful_memes
# ============================================================

import os
import shutil
import kagglehub


PROJECT_ROOT = "/data/deli/MFL-new/MFL-new"
TARGET_DIR = os.path.join(PROJECT_ROOT, "data", "hateful_memes")


def main():
    print("Downloading Hateful Memes dataset from KaggleHub...")

    download_path = kagglehub.dataset_download(
        "williamberrios/hateful-memes"
    )

    print("Downloaded to:", download_path)
    print("Copying to:", TARGET_DIR)

    os.makedirs(TARGET_DIR, exist_ok=True)

    for name in os.listdir(download_path):
        src = os.path.join(download_path, name)
        dst = os.path.join(TARGET_DIR, name)

        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    print("\nDone.")
    print("Files in target directory:")

    for name in os.listdir(TARGET_DIR):
        print(" -", name)


if __name__ == "__main__":
    main()