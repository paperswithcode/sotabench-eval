import numpy as np
from sotabenchapi.client import Client
from sotabenchapi.core import BenchmarkResult, check_inputs
import time

from sotabencheval.utils import calculate_batch_hash, is_server, get_max_memory_allocated
from sotabencheval.semantic_segmentation.utils import ConfusionMatrix


class PASCALVOCEvaluator(object):
    """`PASCAL VOC <https://sotabench.com/benchmarks/semantic-segmentation-on-pascal-voc-2012-val>`_ benchmark.

    Examples:
        Evaluate a FCN model from the torchvision repository:

        .. code-block:: python

            ...

            evaluator = PASCALVOCEvaluator(model_name='FCN ResNet-101', paper_arxiv_id='1605.06211')

            with torch.no_grad():
                for i, (input, target) in enumerate(iterator):
                    ...
                    output = model(input)
                    # output and target should then be flattened into 1D np.ndarrays and passed in below
                    evaluator.update(output=output, target=target)

                    if evaluator.cache_exists:
                        break

            evaluator.save()
    """

    task = "Semantic Segmentation"

    def __init__(self,
                 model_name: str = None,
                 paper_arxiv_id: str = None,
                 paper_pwc_id: str = None,
                 paper_results: dict = None,
                 model_description=None):
        """Initializes a COCO Evaluator object

        Args:
            model_name (str, optional): The name of the model from the
                paper - if you want to link your build to a machine learning
                paper. See the VOC benchmark page for model names,
                https://sotabench.com/benchmarks/semantic-segmentation-on-pascal-voc-2012-val,
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

                    {'Mean IOU': 76.42709, 'Accuracy': 95.31, ...}.

                Ensure that the metric names match those on the sotabench
                leaderboard - for PASCAL VOC it should be 'Mean IOU', 'Accuracy'
            model_description (str, optional): Optional model description.
        """

        # Model metadata

        self.model_name = model_name
        self.paper_arxiv_id = paper_arxiv_id
        self.paper_pwc_id = paper_pwc_id
        self.paper_results = paper_results
        self.model_description = model_description

        self.voc_evaluator = ConfusionMatrix(21)

        self.outputs = np.array([])
        self.targets = np.array([])

        self.results = None

        # Backend variables for hashing and caching

        self.first_batch_processed = False
        self.batch_hash = None
        self.cached_results = False

        # Speed and memory metrics

        self.init_time = time.time()
        self.speed_mem_metrics = {}

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
                        # output and target should then be flattened into 1D np.ndarrays and passed in below
                        evaluator.update(output=output, target=target)

                        if evaluator.cache_exists:
                            break

                evaluator.save()

        :return: bool or None (if not in check mode)
        """
        if not self.first_batch_processed:
            raise ValueError('No batches of data have been processed so no batch_hash exists')

        if not is_server():
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

    def add(self, outputs: np.ndarray, targets: np.ndarray):
        """
        Update the evaluator with new results from the model

        :param outputs (np.ndarray): 1D np.ndarray of semantic class predictions per pixel
        :param targets (np.ndarray): 1D np.ndarray of ground truth semantic classes per pixel

        The method requires an outputs input and a targets input - both flattened.

        Suppose you are making predictions, batch by batch, and have your model outputs
        and the original targets with batch_size 32, and image size 520 x 480.
        The shape of your outputs might look like this:

        batch_output.shape
        >> (32, 21, 520, 480) # where 21 is the number of VOC classes

        batch_target.shape
        >> (32, 520, 480)

        We can flatten the entire output and targets to 1D vectors for each pixel:

        flattened_batch_output.shape
        >> (7987200) # flatten by taking the max class prediction
                     #  (batch_output.argmax(1).flatten() in torch with class as second dimension)

        flattened_batch_target.shape
        >> (7987200) # (batch_target.flatten() in torch)

        The output might look something like this:

        flattened_batch_output
        >> array([6, 6, 6, 6, 6, ...])

        flattened_batch_target
        >> array([6, 6, 6, 6, 6, ...])

        In both cases, the prediction and ground truth have class 6 as the semantic label for the first 5
        pixels - so the model is correct.

        These flattened arrays can then be passed into the .add() method of the evaluator

        .. code-block:: python

            my_evaluator.update(outputs=flattened_batch_output,
                                        targets=flattened_batch_target)


        :return: void - updates self.voc_evaluator with the data, and updates self.targets and self.outputs
        """
        self.voc_evaluator.update(targets, outputs)

        self.targets = np.append(self.targets, targets)
        self.outputs = np.append(self.outputs, outputs)

        if not self.first_batch_processed:
            acc_global, acc, iu = self.voc_evaluator.compute()
            self.batch_hash = calculate_batch_hash(np.append(
                np.append(np.around(targets, 3), np.around(outputs, 3)),
                np.around(np.array([acc_global.item(), iu.mean().item()]), 3)))
            self.first_batch_processed = True

    def get_results(self):
        """
        Reruns the evaluation using the accumulated detections, returns VOC results with IOU and
        Accuracy metrics

        :return: dict with PASCAL VOC metrics
        """
        if self.cached_results:
            return self.results

        self.voc_evaluator = ConfusionMatrix(21)
        self.voc_evaluator.update(self.targets.astype(np.int64), self.outputs.astype(np.int64))

        acc_global, acc, iu = self.voc_evaluator.compute()

        self.results = {
                   "Accuracy": acc_global.item(),
                   "Mean IOU": iu.mean().item(),
               }

        self.speed_mem_metrics['Max Memory Allocated (Total)'] = get_max_memory_allocated()

        return self.results

    def reset_time(self):
        """
        Simple method to reset the timer self.init_time. Often used before a loop, to time the evaluation
        appropriately, for example:

        :return: void - resets self.init_time
        """
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

        # If this is the first time the model is run, then we record evaluation time information

        if not self.cached_results:
            self.speed_mem_metrics['Tasks / Evaluation Time'] = None
            self.speed_mem_metrics['Tasks'] = None
            self.speed_mem_metrics['Evaluation Time'] = (time.time() - self.init_time)
        else:
            self.speed_mem_metrics['Tasks / Evaluation Time'] = None
            self.speed_mem_metrics['Tasks'] = None
            self.speed_mem_metrics['Evaluation Time'] = None

        return BenchmarkResult(
            task=self.task,
            config={},
            dataset='PASCAL VOC 2012 val',
            results=self.results,
            speed_mem_metrics=self.speed_mem_metrics,
            model=self.model_name,
            model_description=self.model_description,
            arxiv_id=self.paper_arxiv_id,
            pwc_id=self.paper_pwc_id,
            paper_results=self.paper_results,
            run_hash=self.batch_hash,
        )
