# Some of the processing logic here is based on the torchvision ImageNet dataset
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/imagenet.py

import numpy as np
import os
import pickle
from sotabenchapi.client import Client
from sotabenchapi.core import BenchmarkResult, check_inputs
import tqdm
import time

from sotabencheval.utils import AverageMeter, calculate_batch_hash, download_url, change_root_if_server, is_server
from sotabencheval.utils import get_max_memory_allocated
from sotabencheval.image_classification.utils import top_k_accuracy_score


ARCHIVE_DICT = {
    'labels': {
        'url': 'https://github.com/paperswithcode/sotabench-eval/releases/download/0.01/imagenet_val_targets.pkl',
        'md5': 'b652e91a9d5b47384fbe8c2e3089778a',
    }
}


class ImageNetEvaluator(object):
    """`ImageNet <https://sotabench.com/benchmarks/image-classification-on-imagenet>`_ benchmark.

    Examples:
        Evaluate a ResNeXt model from the torchvision repository:

        .. code-block:: python

            import numpy as np
            import PIL
            import torch
            from torchvision.models.resnet import resnext101_32x8d
            import torchvision.transforms as transforms
            from torchvision.datasets import ImageNet
            from torch.utils.data import DataLoader

            from sotabencheval.image_classification import ImageNetEvaluator
            from sotabencheval.utils import is_server

            if is_server():
                DATA_ROOT = './.data/vision/imagenet'
            else: # local settings
                DATA_ROOT = '/home/ubuntu/my_data/'

            model = resnext101_32x8d(pretrained=True)

            # Define the transforms need to convert ImageNet data to expected
            # model input
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
            input_transform = transforms.Compose([
                transforms.Resize(256, PIL.Image.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])

            test_dataset = ImageNet(
                DATA_ROOT,
                split="val",
                transform=input_transform,
                target_transform=None,
                download=True,
            )

            test_loader = DataLoader(
                test_dataset,
                batch_size=128,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )

            model = model.cuda()
            model.eval()

            evaluator = ImageNetEvaluator(
                             model_name='ResNeXt-101-32x8d',
                             paper_arxiv_id='1611.05431')

            with torch.no_grad():
                for i, (input, target) in enumerate(test_loader):
                    input = input.to(device='cuda', non_blocking=True)
                    target = target.to(device='cuda', non_blocking=True)
                    output = model(input)

                    image_ids = [img[0].split('/')[-1].replace('.JPEG', '') for img in test_loader.dataset.imgs[i*test_loader.batch_size:(i+1)*test_loader.batch_size]]

                    evaluator.add(dict(zip(image_ids, list(output.cpu().numpy()))))

                    if evaluator.cache_exists:
                        break

            print(evaluator.get_results()) # print results of evaluation

            evaluator.save()
    """

    task = "Image Classification"

    def __init__(self,
                 root: str = '.',
                 model_name: str = None,
                 paper_arxiv_id: str = None,
                 paper_pwc_id: str = None,
                 paper_results: dict = None,
                 model_description=None,):
        """Initializes an ImageNet Evaluator object

        Args:
            root (string): Root directory of the ImageNet Dataset - where the
            label data is located (or will be downloaded to). Note this does not download
            the full ImageNet dataset (!) but just annotation information.
            model_name (str, optional): The name of the model from the
                paper - if you want to link your build to a model from a
                machine learning paper. See the ImageNet benchmark page for model names,
                https://sotabench.com/benchmarks/image-classification-on-imagenet, e.g.
                on the paper leaderboard tab.
            paper_arxiv_id (str, optional): Optional linking to arXiv if you
                want to link to papers on the leaderboard; put in the
                corresponding paper's arXiv ID, e.g. '1611.05431'.
            paper_pwc_id (str, optional): Optional linking to Papers With Code;
                put in the corresponding papers with code URL slug, e.g.
                'u-gat-it-unsupervised-generative-attentional'
            paper_results (dict, optional) : If the paper model you are reproducing
                does not have model results on sotabench.com, you can specify
                the paper results yourself through this argument, where keys
                are metric names, values are metric values. e.g::

                    {'Top 1 Accuracy': 0.543, 'Top 5 Accuracy': 0.654}.

                Ensure that the metric names match those on the sotabench
                leaderboard - for ImageNet it should be 'Top 1 Accuracy' and
                'Top 5 Accuracy'.
            model_description (str, optional): Optional model description.
        """
        root = self.root = os.path.expanduser(change_root_if_server(
            root=root,
            server_root="./.data/vision/imagenet"))

        # Model metadata

        self.model_name = model_name
        self.paper_arxiv_id = paper_arxiv_id
        self.paper_pwc_id = paper_pwc_id
        self.paper_results = paper_results
        self.model_description = model_description

        self.top1 = AverageMeter()
        self.top5 = AverageMeter()
        self.load_targets()

        self.outputs = {}
        self.results = None

        # Backend variables for hashing and caching

        self.first_batch_processed = False
        self.batch_hash = None
        self.cached_results = False

        # Speed and memory metrics

        self.speed_mem_metrics = {}
        self.init_time = time.time()

    @property
    def cache_exists(self):
        """
        Checks whether the cache exists in the sotabench.com database - if so
        then sets self.results to cached results and returns True.

        You can use this property for control flow to break a for loop over a dataset
        after the first iteration. This prevents re-running the same calculation for the
        same model twice.

        Q: Why should the user use this?
        A: If you want fast "continuous evaluation" and don't want to avoid rerunning the same model over and over
            each time you commit something new to your repository.

        Examples:
            Breaking a for loop for a PyTorch evaluation

            .. code-block:: python

                ...

                with torch.no_grad():
                    for i, (input, target) in enumerate(test_loader):
                        input = input.to(device=device, non_blocking=True)
                        target = target.to(device=device, non_blocking=True)
                        output = model(input)

                        image_ids = [img[0].split('/')[-1].replace('.JPEG', '') for img in test_loader.dataset.imgs[i*test_loader.batch_size:(i+1)*test_loader.batch_size]]

                        evaluator.add(dict(zip(image_ids, list(output.cpu().numpy()))))

                        if evaluator.cache_exists:
                            break

                evaluator.save()  # uses the cached results

        This logic is for the server; it will not break the loop if you evaluate locally.

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

    def load_targets(self):
        """
        Downloads ImageNet labels and IDs and puts into self.root, then loads to self.targets

        :return: void - update self.targets with the ImageNet validation data labels, and downloads if
        the pickled validation data is not in the root location
        """
        download_url(
            url=ARCHIVE_DICT['labels']['url'],
            root=self.root,
            md5=ARCHIVE_DICT['labels']['md5'])

        with open(os.path.join(self.root, 'imagenet_val_targets.pkl'), 'rb') as handle:
            self.targets = pickle.load(handle)

    def add(self, output_dict: dict):
        """
        Updates the evaluator with new results

        :param output_dict: (dict) Where keys are image IDs, and each value should be an 1D np.ndarray of size 1000
        containing logits for that image ID.
        :return: void - updates self.outputs with the new IDSs and prediction

        Examples:
            Update the evaluator with two results:

            .. code-block:: python

                my_evaluator.add({'ILSVRC2012_val_00000293': np.array([1.04243, ...]),
                'ILSVRC2012_val_00000294': np.array([-2.3677, ...])})
        """
        if not output_dict:
            print('Empty output_dict; will not process')
            return

        self.outputs = dict(list(self.outputs.items()) + list(output_dict.items()))

        for i, dict_key in enumerate(output_dict.keys()):
            output = self.outputs[dict_key]
            target = self.targets[dict_key]
            prec1 = top_k_accuracy_score(y_true=target, y_pred=np.array([output]), k=1)
            prec5 = top_k_accuracy_score(y_true=target, y_pred=np.array([output]), k=5)
            self.top1.update(prec1, 1)
            self.top5.update(prec5, 1)

        if not self.first_batch_processed:
            hash_dict = output_dict
            hash_dict['Top 1 Accuracy'] = self.top1.avg
            hash_dict['Top 5 Accuracy'] = self.top5.avg
            self.batch_hash = calculate_batch_hash(hash_dict)
            self.first_batch_processed = True

    def get_results(self):
        """
        Gets the results for the evaluator. This method only runs if predictions for all 5,000 ImageNet validation
        images are available. Otherwise raises an error and informs you of the missing or unmatched IDs.

        :return: dict with Top 1 and Top 5 Accuracy
        """

        if self.cached_results:
            return self.results

        if set(self.targets.keys()) != set(self.outputs.keys()):
            missing_ids = set(self.targets.keys()) - set(self.outputs.keys())
            unmatched_ids = set(self.outputs.keys()) - set(self.targets.keys())

            if len(unmatched_ids) > 0:
                raise ValueError('''There are {mis_no} missing and {un_no} unmatched image IDs\n\n'''
                                     '''Missing IDs are {missing}\n\n'''
                                     '''Unmatched IDs are {unmatched}'''.format(mis_no=len(missing_ids),
                                                                                un_no=len(unmatched_ids),
                                                                                missing=missing_ids,
                                                                                unmatched=unmatched_ids))
            else:
                raise ValueError('''There are {mis_no} missing image IDs\n\n'''
                                     '''Missing IDs are {missing}'''.format(mis_no=len(missing_ids),
                                                                            missing=missing_ids))

        # Do the calculation only if we have all the results...
        self.top1 = AverageMeter()
        self.top5 = AverageMeter()

        for i, dict_key in enumerate(tqdm.tqdm(self.targets.keys())):
            output = self.outputs[dict_key]
            target = self.targets[dict_key]
            prec1 = top_k_accuracy_score(y_true=target, y_pred=np.array([output]), k=1)
            prec5 = top_k_accuracy_score(y_true=target, y_pred=np.array([output]), k=5)
            self.top1.update(prec1, 1)
            self.top5.update(prec5, 1)

        self.results = {'Top 1 Accuracy': self.top1.avg, 'Top 5 Accuracy': self.top5.avg}
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

        On the sotabench.com server, this will produce a JSON file serialisation in sotabench_results.json and results
        will be recorded on the platform.

        :return: BenchmarkResult object with results and metadata
        """
        # recalculate to ensure no mistakes made during batch-by-batch metric calculation
        self.get_results()

        # If this is the first time the model is run, then we record evaluation time information

        if not self.cached_results:
            exec_speed = (time.time() - self.init_time)
            self.speed_mem_metrics['Tasks / Evaluation Time'] = len(self.outputs) / exec_speed
            self.speed_mem_metrics['Tasks'] = len(self.outputs)
            self.speed_mem_metrics['Evaluation Time'] = exec_speed
        else:
            self.speed_mem_metrics['Tasks / Evaluation Time'] = None
            self.speed_mem_metrics['Tasks'] = None
            self.speed_mem_metrics['Evaluation Time'] = None

        return BenchmarkResult(
            task=self.task,
            config={},
            dataset='ImageNet',
            results=self.results,
            speed_mem_metrics=self.speed_mem_metrics,
            model=self.model_name,
            model_description=self.model_description,
            arxiv_id=self.paper_arxiv_id,
            pwc_id=self.paper_pwc_id,
            paper_results=self.paper_results,
            run_hash=self.batch_hash,
        )
