# Code is based on  https://github.com/pytorch/vision/blob/master/references/detection/

import numpy as np
import copy

from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import pycocotools.mask as mask_util

from collections import defaultdict


class CocoEvaluator(object):
    """
    For now this only does BBOX detection - so 'bbox' is the only acceptable iou_type
    """
    def __init__(self, coco_gt, iou_types):
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt

        self.iou_types = iou_types
        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)

        self.annotation_list = []

    def update(self, annotation_list):
        assert(type(annotation_list) == list)

        self.annotation_list.extend(annotation_list)

        for iou_type in self.iou_types:
            coco_dt = loadRes(self.coco_gt, self.annotation_list) if self.annotation_list else COCO()
            coco_eval = self.coco_eval[iou_type]
            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = self.coco_gt.getImgIds()

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def evaluate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.evaluate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            # print("IoU metric: {}".format(iou_type))
            coco_eval.summarize()


#################################################################
# From pycocotools, just removed the prints and fixed
# a Python3 bug about unicode not defined
#################################################################

# Ideally, pycocotools wouldn't have hard-coded prints
# so that we could avoid copy-pasting those two functions


def createIndex(self):
    # create index
    # print('creating index...')
    anns, cats, imgs = {}, {}, {}
    imgToAnns, catToImgs = defaultdict(list), defaultdict(list)
    if "annotations" in self.dataset:
        for ann in self.dataset["annotations"]:
            imgToAnns[ann["image_id"]].append(ann)
            anns[ann["id"]] = ann

    if "images" in self.dataset:
        for img in self.dataset["images"]:
            imgs[img["id"]] = img

    if "categories" in self.dataset:
        for cat in self.dataset["categories"]:
            cats[cat["id"]] = cat

    if "annotations" in self.dataset and "categories" in self.dataset:
        for ann in self.dataset["annotations"]:
            catToImgs[ann["category_id"]].append(ann["image_id"])

    # print('index created!')

    # create class members
    self.anns = anns
    self.imgToAnns = imgToAnns
    self.catToImgs = catToImgs
    self.imgs = imgs
    self.cats = cats


maskUtils = mask_util


def loadRes(coco, anns):
    """Load result file and return a result api object.

    ``anns`` is a list of dicts containing the results

    In the original pycoco api, a results file is passed in, whereas in this
    case we bypass the json file loading and ask for a list of dictionary
    annotations to be passed directly in

    Returns:
        res (obj): result api object.
    """
    res = COCO()
    res.dataset["images"] = [img for img in coco.dataset["images"]]

    # print('Loading and preparing results...')
    # tic = time.time()
    # if isinstance(resFile, torch._six.string_classes):
    #     anns = json.load(open(resFile))
    # elif type(resFile) == np.ndarray:
    #     anns = self.loadNumpyAnnotations(resFile)
    # else:
    #     anns = resFile
    assert type(anns) == list, "results in not an array of objects"
    annsImgIds = [ann["image_id"] for ann in anns]
    assert set(annsImgIds) == (
        set(annsImgIds) & set(coco.getImgIds())
    ), "Results do not correspond to current coco set"
    if "caption" in anns[0]:
        imgIds = set([img["id"] for img in res.dataset["images"]]) & set(
            [ann["image_id"] for ann in anns]
        )
        res.dataset["images"] = [
            img for img in res.dataset["images"] if img["id"] in imgIds
        ]
        for id, ann in enumerate(anns):
            ann["id"] = id + 1
    elif "bbox" in anns[0] and not anns[0]["bbox"] == []:
        res.dataset["categories"] = copy.deepcopy(coco.dataset["categories"])
        for id, ann in enumerate(anns):
            bb = ann["bbox"]
            x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]
            if "segmentation" not in ann:
                ann["segmentation"] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
            ann["area"] = bb[2] * bb[3]
            ann["id"] = id + 1
            ann["iscrowd"] = 0
    elif "segmentation" in anns[0]:
        res.dataset["categories"] = copy.deepcopy(coco.dataset["categories"])
        for id, ann in enumerate(anns):
            # now only support compressed RLE format as segmentation results
            ann["area"] = maskUtils.area(ann["segmentation"])
            if "bbox" not in ann:
                ann["bbox"] = maskUtils.toBbox(ann["segmentation"])
            ann["id"] = id + 1
            ann["iscrowd"] = 0
    elif "keypoints" in anns[0]:
        res.dataset["categories"] = copy.deepcopy(coco.dataset["categories"])
        for id, ann in enumerate(anns):
            s = ann["keypoints"]
            x = s[0::3]
            y = s[1::3]
            x0, x1, y0, y1 = np.min(x), np.max(x), np.min(y), np.max(y)
            ann["area"] = (x1 - x0) * (y1 - y0)
            ann["id"] = id + 1
            ann["bbox"] = [x0, y0, x1 - x0, y1 - y0]
    # print('DONE (t={:0.2f}s)'.format(time.time()- tic))

    res.dataset["annotations"] = anns
    createIndex(res)
    return res
