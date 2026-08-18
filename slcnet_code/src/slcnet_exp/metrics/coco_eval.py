from __future__ import annotations

import argparse
from pathlib import Path

from slcnet_exp.common.io import write_json


def evaluate_coco(gt_json: str | Path, pred_json: str | Path) -> dict[str, float]:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise RuntimeError("pycocotools is required for COCO mAP evaluation.") from exc
    coco_gt = COCO(str(gt_json))
    coco_dt = coco_gt.loadRes(str(pred_json))
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return {
        "mAP_0.5_0.95": float(evaluator.stats[0]),
        "mAP_0.5": float(evaluator.stats[1]),
        "mAP_0.75": float(evaluator.stats[2]),
        "AP_small": float(evaluator.stats[3]),
        "AP_medium": float(evaluator.stats[4]),
        "AP_large": float(evaluator.stats[5]),
        "Recall_small": float(evaluator.stats[9]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate COCO bbox predictions.")
    parser.add_argument("--gt", required=True, help="Ground-truth COCO JSON.")
    parser.add_argument("--pred", required=True, help="Prediction COCO result JSON.")
    parser.add_argument("--out", default="reports/coco_eval.json", help="Output metrics JSON.")
    args = parser.parse_args()
    metrics = evaluate_coco(args.gt, args.pred)
    write_json(args.out, metrics)
    print(metrics)


if __name__ == "__main__":
    main()
