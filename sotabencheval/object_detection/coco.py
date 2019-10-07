# Some of the processing logic here is based on the torchvision COCO dataset

import copy
import numpy as np
import os
from pycocotools.coco import COCO
from sotabenchapi.client import Client
from sotabenchapi.core import BenchmarkResult, check_inputs
import time

from sotabencheval.utils import calculate_batch_hash, extract_archive, change_root_if_server, is_server
from sotabencheval.utils import get_max_memory_allocated
from sotabencheval.object_detection.coco_eval import CocoEvaluator
from sotabencheval.object_detection.utils import get_coco_metrics


class COCOEvaluator(object):
    """`COCO <https://sotabench.com/benchmarks/object-detection-on-coco-minival>`_ benchmark.

    Examples:
        Evaluate a ResNeXt model from the torchvision repository:

        .. code-block:: python

            ...

            evaluator = COCOEvaluator(model_name='Mask R-CNN', paper_arxiv_id='1703.06870')

            with torch.no_grad():
                for i, (input, __) in enumerate(iterator):
                    ...
                    output = model(input)
                    # optional formatting of output here to be a list of detection dicts
                    evaluator.add(output)

                    if evaluator.cache_exists:
                        break

            evaluator.save()
    """

    task = "Object Detection"

    def __init__(self,
                 root: str = '.',
                 split: str = "val",
                 dataset_year: str = "2017",
                 model_name: str = None,
                 paper_arxiv_id: str = None,
                 paper_pwc_id: str = None,
                 paper_results: dict = None,
                 model_description=None,):
        """Benchmarking function.

        Args:
            root (string): Root directory of the COCO Dataset - where the
            label data is located (or will be downloaded to).
            split (str) : the split for COCO to use, e.g. 'val'
            dataset_year (str): the dataset year for COCO to use
            model_name (str, optional): The name of the model from the
                paper - if you want to link your build to a machine learning
                paper. See the COCO benchmark page for model names,
                https://sotabench.com/benchmarks/object-detection-on-coco-minival,
                e.g. on the paper leaderboard tab.
            paper_arxiv_id (str, optional): Optional linking to arXiv if you
                want to link to papers on the leaderboard; put in the
                corresponding paper's arXiv ID, e.g. '1611.05431'.
            paper_pwc_id (str, optional): Optional linking to Papers With Code;
                put in the corresponding papers with code URL slug, e.g.
                'u-gat-it-unsupervised-generative-attentional'
            paper_results (dict, optional) : If the paper you are reproducing
                does not have model results on sotabench.com, you can specify
                the paper results yourself through this argument, where keys
                are metric names, values are metric values. e.g::

                    {'box AP': 0.349, 'AP50': 0.592, ...}.

                Ensure that the metric names match those on the sotabench
                leaderboard - for COCO it should be 'box AP', 'AP50',
                'AP75', 'APS', 'APM', 'APL'
            model_description (str, optional): Optional model description.
        """

        root = self.root = change_root_if_server(root=root,
                                                 server_root="./.data/vision/coco")

        self.model_name = model_name
        self.paper_arxiv_id = paper_arxiv_id
        self.paper_pwc_id = paper_pwc_id
        self.paper_results = paper_results
        self.model_description = model_description
        self.split = split

        annFile = os.path.join(
            root, "annotations/instances_%s%s.json" % (self.split, dataset_year)
        )

        self._download(annFile)

        self.coco = COCO(annFile)
        self.iou_types = ['bbox']
        self.coco_evaluator = CocoEvaluator(self.coco, self.iou_types)

        self.detections = []
        self.results = None
        self.first_batch_processed = False
        self.batch_hash = None
        self.cached_results = False

        self.speed_mem_metrics = {}

        self.init_time = time.time()

    def _download(self, annFile):
        if not os.path.isdir(annFile):
            if "2017" in annFile:
                annotations_dir_zip = os.path.join(
                    self.root, "annotations_train%s2017.zip" % self.split
                )
            elif "2014" in annFile:
                annotations_dir_zip = os.path.join(
                    self.root, "annotations_train%s2014.zip" % self.split
                )
            else:
                annotations_dir_zip = None

            if annotations_dir_zip is not None:
                print('Attempt to extract annotations file at {zip_loc}'.format(zip_loc=annotations_dir_zip))
                extract_archive(from_path=annotations_dir_zip, to_path=self.root)

    @property
    def cache_exists(self):
        """
        Checks whether the cache exists in the sotabench.com database - if so
        then sets self.results to cached results and returns True.

        You can use this property for control flow to break a for loop over a dataset
        after the first iteration. This prevents rerunning the same calculation for the
        same model twice.

        Examples:
            Breaking a for loop

            .. code-block:: python

                ...

                with torch.no_grad():
                    for i, (input, target) in enumerate(iterator):
                        ...
                        output = model(input)
                        # optional formatting of output here to be a list of detection dicts
                        evaluator.add(output)

                        if evaluator.cache_exists:
                            break

                evaluator.save()

        :return: bool or None (if not in check mode)
        """

        if not self.first_batch_processed:
            raise ValueError('No batches of data have been processed so no batch_hash exists')

        if not is_server():  # we only check the cache on the server
            return None

        client = Client.public()
        cached_res = client.get_results_by_run_hash(self.batch_hash)
        if cached_res:
            self.results = cached_res
            self.cached_results = True
            print(
                "No model change detected (using the first batch run "
                "hash). Will use cached results."
            )
            return True

        return False

    @staticmethod
    def cache_format_ann(ann):
        """
        Cache formats an annotation dictionary with rounding. the reason we need to round is that if we have
        small floating point originated differences, then changes the hash of the predictions.

        :param ann (dict): A detection dictionary

        :return: ann : A detection dictionary but with rounded values
        """
        ann['bbox'] = [np.round(el, 3) for el in ann['bbox']]
        ann['score'] = np.round(ann['score'], 3)

        if 'segmentation' in ann:
            ann['segmentation'] = [np.round(el, 3) for el in ann['segmentation']]

        if 'area' in ann:
            ann['area'] = np.round(ann['area'], 3)

        return ann

    def cache_values(self, annotations, metrics):
        """
        Takes in annotations and metrics, and formats the data to calculate the hash for the cache
        :param annotations: list of detections
        :param metrics: dictionary of final AP metrics
        :return: list of data (combining annotations and metrics)
        """

        metrics = {k: np.round(v, 3) for k, v in metrics.items()}
        new_annotations = copy.deepcopy(annotations)
        new_annotations = [self.cache_format_ann(ann) for ann in new_annotations]

        return new_annotations + [metrics]

    def add(self, detections: list):
        """
        Update the evaluator with new detections

        :param annotations (list): List of detections, that will be used by the COCO.loadRes method in the
        pycocotools API.  Each detection can take a dictionary format like the following:

        {'image_id': 397133, 'bbox': [386.1628112792969, 69.48855590820312, 110.14895629882812, 278.2847595214844],
        'score': 0.999152421951294, 'category_id': 1}

        I.e is a list of dictionaries.

        :return: void - updates self.detection with the new IDSs and prediction

        Examples:
            Update the evaluator with two results:

            .. code-block:: python

                my_evaluator.add([{'image_id': 397133, 'bbox': [386.1628112792969, 69.48855590820312,
                110.14895629882812, 278.2847595214844], 'score': 0.999152421951294, 'category_id': 1}])
        """

        self.detections.extend(detections)

        self.coco_evaluator.update(detections)

        if not self.first_batch_processed:
            self.coco_evaluator.evaluate()
            self.coco_evaluator.accumulate()

            if any([detection['bbox'] for detection in detections]): # we can only hash if we have predictions
                self.batch_hash = calculate_batch_hash(
                    self.cache_values(annotations=detections, metrics=get_coco_metrics(self.coco_evaluator)))
                self.first_batch_processed = True

    def get_results(self):
        """
        Reruns the evaluation using the accumulated detections, returns COCO results with AP metrics

        :return: dict with COCO AP metrics
        """

        if self.cached_results:
            return self.results

        self.coco_evaluator = CocoEvaluator(self.coco, self.iou_types)
        self.coco_evaluator.update(self.detections)
        self.coco_evaluator.evaluate()
        self.coco_evaluator.accumulate()
        self.coco_evaluator.summarize()

        self.results = get_coco_metrics(self.coco_evaluator)
        self.speed_mem_metrics['Max Memory Allocated (Total)'] = get_max_memory_allocated()

        return self.results

    def reset_time(self):
        self.init_time = time.time()

    def save(self):
        """
        Calculate results and then put into a BenchmarkResult object

        On the sotabench.com server, this will produce a JSON file serialisation and results will be recorded
        on the platform.

        :return: BenchmarkResult object with results and metadata
        """

        # recalculate to ensure no mistakes made during batch-by-batch metric calculation
        self.get_results()

        if not self.cached_results:
            unique_image_ids = set([d['image_id'] for d in self.detections])
            exec_speed = (time.time() - self.init_time)
            self.speed_mem_metrics['Tasks / Evaluation Time'] = len(unique_image_ids) / exec_speed
            self.speed_mem_metrics['Tasks'] = len(unique_image_ids)
            self.speed_mem_metrics['Evaluation Time'] = exec_speed
        else:
            self.speed_mem_metrics['Tasks / Evaluation Time'] = None
            self.speed_mem_metrics['Tasks'] = None
            self.speed_mem_metrics['Evaluation Time'] = None

        return BenchmarkResult(
            task=self.task,
            config={},
            dataset='COCO minival',
            results=self.results,
            speed_mem_metrics=self.speed_mem_metrics,
            model=self.model_name,
            model_description=self.model_description,
            arxiv_id=self.paper_arxiv_id,
            pwc_id=self.paper_pwc_id,
            paper_results=self.paper_results,
            run_hash=self.batch_hash,
        )
