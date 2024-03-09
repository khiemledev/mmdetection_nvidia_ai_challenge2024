import json
import shutil
from argparse import ArgumentParser
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def build_args():
    parser = ArgumentParser()
    parser.add_argument("json_path", type=str, help="json file")
    parser.add_argument("--output", type=str, default="tfboard_result")
    return parser.parse_args()


def main():
    args = build_args()

    jsonl_file = Path(args.json_path)
    log_dir = Path("tfboard_result_" + jsonl_file.stem)
    shutil.rmtree(log_dir, ignore_errors=True)

    # Create a SummaryWriter object
    writer = SummaryWriter(log_dir=log_dir)

    train_metrics = [
        "lr",
        # "loss",
        "loss_cls",
        "loss_bbox",
        "loss_iou",
        "epoch",
    ]
    val_metrics = [
        "coco/bbox_mAP",
        "coco/bbox_mAP_50",
        "coco/bbox_mAP_75",
        "coco/bbox_mAP_s",
        "coco/bbox_mAP_m",
        "coco/bbox_mAP_l",
        # "data_time",
        # "time",
        "step",
    ]

    with jsonl_file.open("r") as f:
        last_datal = None
        for l in f:
            datal = json.loads(l.strip())

            if not "coco/bbox_mAP" in datal:
                for k in train_metrics:
                    if last_datal:
                        writer.add_scalar(
                            f"train/{k}",
                            last_datal.get(k, 0),
                            last_datal.get("epoch"),
                        )
                last_datal = datal
            else:
                # this is val metrics
                for k in val_metrics:
                    writer.add_scalar(
                        f"val/{k}",
                        datal.get(k, 0),
                        datal.get("step"),
                    )

    # Close the writer when you're done
    writer.close()

    print(f"Result saved to {log_dir}")

if __name__ == "__main__":
    main()
