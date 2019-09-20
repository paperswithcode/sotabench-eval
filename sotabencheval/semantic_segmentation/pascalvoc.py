import numpy as np
import os
from sotabenchapi.check import in_check_mode
from sotabenchapi.client import Client
from sotabenchapi.core import BenchmarkResult, check_inputs
import tarfile

from sotabencheval.utils import calculate_batch_hash, download_url, change_root_if_server
from sotabencheval.semantic_segmentation.utils import ConfusionMatrix

DATASET_YEAR_DICT = {
    '2012': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
        'filename': 'VOCtrainval_11-May-2012.tar',
        'md5': '6cd6e144f989b92b3379bac3b3de84fd',
        'base_dir': 'VOCdevkit/VOC2012'
    },
    '2011': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2011/VOCtrainval_25-May-2011.tar',
        'filename': 'VOCtrainval_25-May-2011.tar',
        'md5': '6c3384ef61512963050cb5d687e5bf1e',
        'base_dir': 'TrainVal/VOCdevkit/VOC2011'
    },
    '2010': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar',
        'filename': 'VOCtrainval_03-May-2010.tar',
        'md5': 'da459979d0c395079b5c75ee67908abb',
        'base_dir': 'VOCdevkit/VOC2010'
    },
    '2009': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2009/VOCtrainval_11-May-2009.tar',
        'filename': 'VOCtrainval_11-May-2009.tar',
        'md5': '59065e4b188729180974ef6572f6a212',
        'base_dir': 'VOCdevkit/VOC2009'
    },
    '2008': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2008/VOCtrainval_14-Jul-2008.tar',
        'filename': 'VOCtrainval_11-May-2012.tar',
        'md5': '2629fa636546599198acfcfbfcf1904a',
        'base_dir': 'VOCdevkit/VOC2008'
    },
    '2007': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
        'filename': 'VOCtrainval_06-Nov-2007.tar',
        'md5': 'c52e279531787c972589f7e41ab4ae64',
        'base_dir': 'VOCdevkit/VOC2007'
    }
}


class PASCALVOCEvaluator(object):
    """`PASCAL VOC <https://www.sotabench.com/benchmark/pascalvoc2012>`_ benchmark.

    Examples:
        Evaluate a FCN model from the torchvision repository:

        .. code-block:: python

            ...

            evaluator = PASCALVOCEvaluator(
                             paper_model_name='FCN ResNet-101',
                             paper_arxiv_id='1605.06211')

            with torch.no_grad():
                for i, (input, target) in enumerate(iterator):
                    input, target = send_data_to_device(input, target, device=device)
                    output = model(input)
                    output, target = model_output_transform(output, target)

                    evaluator.update(output=output, target=target)

            evaluator.save()
    """

    task = "Semantic Segmentation"

    def __init__(self,
                 root: str = '.',
                 split: str = "val",
                 dataset_year: str = "2012",
                 paper_model_name: str = None,
                 paper_arxiv_id: str = None,
                 paper_pwc_id: str = None,
                 paper_results: dict = None,
                 pytorch_hub_url: str = None,
                 model_description=None,
                 download: bool = False):
        """Benchmarking function.

        Args:
            root (string): Root directory of the PASCAL VOC Dataset.
            split (str) : the split for PASCAL VOC to use, e.g. 'val'
            dataset_year (str): the dataset year for PASCAL VOC to use
            paper_model_name (str, optional): The name of the model from the
                paper - if you want to link your build to a machine learning
                paper. See the VOC benchmark page for model names,
                https://www.sotabench.com/benchmark/pascalvoc2012, e.g. on the
                paper leaderboard tab.
            paper_arxiv_id (str, optional): Optional linking to ArXiv if you
                want to link to papers on the leaderboard; put in the
                corresponding paper's ArXiv ID, e.g. '1611.05431'.
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
            pytorch_hub_url (str, optional): Optional linking to PyTorch Hub
                url if your model is linked there; e.g:
                'nvidia_deeplearningexamples_waveglow'.
            model_description (str, optional): Optional model description.
            download (bool) : whether to download the data or not
        """

        root = self.root = change_root_if_server(root=root,
                                                 server_root="./.data/vision/voc%s" % dataset_year)

        self.paper_model_name = paper_model_name
        self.paper_arxiv_id = paper_arxiv_id
        self.paper_pwc_id = paper_pwc_id
        self.paper_results = paper_results
        self.pytorch_hub_url = pytorch_hub_url
        self.model_description = model_description
        self.dataset_year = dataset_year
        self.split = split

        self.voc_evaluator = ConfusionMatrix(21)

        self.outputs = np.array([])
        self.targets = np.array([])

        self.url = DATASET_YEAR_DICT[self.dataset_year]['url']
        self.filename = DATASET_YEAR_DICT[self.dataset_year]['filename']
        self.md5 = DATASET_YEAR_DICT[self.dataset_year]['md5']

        base_dir = DATASET_YEAR_DICT[self.dataset_year]['base_dir']
        voc_root = os.path.join(self.root, base_dir)
        image_dir = os.path.join(voc_root, 'JPEGImages')
        mask_dir = os.path.join(voc_root, 'SegmentationClass')

        if download:
            download_extract(self.url, self.root, self.filename, self.md5)

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        splits_dir = os.path.join(voc_root, 'ImageSets/Segmentation')

        split_f = os.path.join(splits_dir, self.split.rstrip('\n') + '.txt')

        with open(os.path.join(split_f), "r") as f:
            self.file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in self.file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in self.file_names]

        assert (len(self.images) == len(self.masks))

        self.results = None
        self.first_batch_processed = False
        self.batch_hash = None

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
                        input, target = send_data_to_device(input, target, device=device)
                        original_output = model(input)
                        output, target = model_output_transform(original_output, target)
                        result = {
                            tar["image_id"].item(): out for tar, out in zip(target, output)
                        }
                        result = prepare_for_coco_detection(result) # convert to right format

                        evaluator.add(result)

                        if evaluator.cache_exists:
                            break

                evaluator.save()

        :return: bool or None (if not in check mode)
        """

        if not self.first_batch_processed:
            raise ValueError('No batches of data have been processed so no batch_hash exists')

        if not in_check_mode():
            return None

        client = Client.public()
        cached_res = client.get_results_by_run_hash(self.batch_hash)
        if cached_res:
            self.results = cached_res
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
            self.batch_hash = calculate_batch_hash(targets + outputs)
            self.first_batch_processed = True

    def get_results(self):
        """
        Reruns the evaluation using the accumulated detections, returns VOC results with IOU and
        Accuracy metrics

        :return: dict with PASCAL VOC metrics
        """

        self.voc_evaluator = ConfusionMatrix(21)
        self.voc_evaluator.update(self.targets.astype(np.int64), self.outputs.astype(np.int64))

        acc_global, acc, iu = self.voc_evaluator.compute()

        self.results = {
                   "Accuracy": acc_global.item() * 100,
                   "Mean IOU": iu.mean().item() * 100,
               }

        return self.results

    def save(self):
        """
        Calculate results and then put into a BenchmarkResult object

        On the sotabench.com server, this will produce a JSON file serialisation and results will be recorded
        on the platform.

        :return: BenchmarkResult object with results and metadata
        """

        # recalculate to ensure no mistakes made during batch-by-batch metric calculation
        self.get_results()

        return BenchmarkResult(
            task=self.task,
            config={},
            dataset='PASCAL VOC %s' % self.dataset_year,
            results=self.results,
            pytorch_hub_id=self.pytorch_hub_url,
            model=self.paper_model_name,
            model_description=self.model_description,
            arxiv_id=self.paper_arxiv_id,
            pwc_id=self.paper_pwc_id,
            paper_results=self.paper_results,
            run_hash=self.batch_hash,
        )


def download_extract(url, root, filename, md5):
    download_url(url, root, filename, md5)
    with tarfile.open(os.path.join(root, filename), "r") as tar:
        tar.extractall(path=root)
