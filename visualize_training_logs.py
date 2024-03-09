import json
from pathlib import Path
import shutil

from torch.utils.tensorboard import SummaryWriter


def main():
    # jsonl_file = Path("exp1/20240306_154903/vis_data/20240306_154903.json")
    # log_dir = Path("tfboard_result_20240306_154903")
    jsonl_file = Path("cloud_train/mmdetection_nvidia_ai_challenge2024/exp2/20240307_113802/vis_data/20240307_113802.json")
    log_dir = Path("tfboard_result_20240307_113802")
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

