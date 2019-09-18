import numpy as np

def get_coco_metrics(coco_evaluator):

    metrics = {
        "box AP": None,
        "AP50": None,
        "AP75": None,
        "APS": None,
        "APM": None,
        "APL": None,
    }
    iouThrs = [None, 0.5, 0.75, None, None, None]
    maxDets = [100] + [coco_evaluator.coco_eval["bbox"].params.maxDets[2]] * 5
    areaRngs = ["all", "all", "all", "small", "medium", "large"]
    bounding_box_params = coco_evaluator.coco_eval["bbox"].params

    for metric_no, metric in enumerate(metrics):
        aind = [
            i
            for i, aRng in enumerate(bounding_box_params.areaRngLbl)
            if aRng == areaRngs[metric_no]
        ]
        mind = [
            i
            for i, mDet in enumerate(bounding_box_params.maxDets)
            if mDet == maxDets[metric_no]
        ]

        s = coco_evaluator.coco_eval["bbox"].eval["precision"]

        # IoU
        if iouThrs[metric_no] is not None:
            t = np.where(iouThrs[metric_no] == bounding_box_params.iouThrs)[0]
            s = s[t]
        s = s[:, :, :, aind, mind]

        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])

        metrics[metric] = mean_s

    return metrics
