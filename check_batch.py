import yaml
from src.utils import load_json
from src.federated import build_dataloader


class DummyTokenizer:
    def __call__(
        self,
        text,
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt",
    ):
        import torch
        return {
            "input_ids": torch.zeros(1, max_length, dtype=torch.long),
            "attention_mask": torch.zeros(1, max_length, dtype=torch.long),
        }


class Args:
    pass


with open("configs/config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

args = Args()
args.max_text_len = cfg["evaluation"]["max_text_len"]
args.out_dir = "results/debug_check"
args.image_model_name = cfg["model"]["image_model_name"]
args.batch_size = 8
args.num_workers = 0

data = load_json(cfg["data"]["train_json"])

loader = build_dataloader(
    samples=data[:32],
    tokenizer=DummyTokenizer(),
    args=args,
    mode="image",
    shuffle=False,
)

batch = next(iter(loader))

img = batch["pixel_values"] if "pixel_values" in batch else batch["image"]

print("image shape:", img.shape)
print("image min:", img.min().item())
print("image max:", img.max().item())
print("image mean:", img.mean().item())
print("image std:", img.std().item())
print("labels:", batch["label"].tolist())