import numpy as np
import os
import pickle
from sotabenchapi.core import BenchmarkResult, check_inputs
import tqdm

from sotabencheval.utils import AverageMeter, download_url
from .utils import top_k_accuracy_score

ARCHIVE_DICT = {
    'labels': {
        'url': 'https://github.com/paperswithcode/sotabench-eval/releases/download/0.01/imagenet_val_targets.pkl',
        'md5': 'b652e91a9d5b47384fbe8c2e3089778a',
    }
}


class ImageNetEvaluator(object):
    """`ImageNet <https://www.sotabench.com/benchmark/imagenet>`_ benchmark.

    Examples:
        Evaluate a ResNeXt model from the torchvision repository:

        .. code-block:: python

            import numpy as np
            import PIL
            import torch
            from sotabencheval.image_classification import ImageNetEvaluator
            from torchvision.models.resnet import resnext101_32x8d
            import torchvision.transforms as transforms
            from torchvision.datasets import ImageNet
            from torch.utils.data import DataLoader

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
                './data',
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

            final_output = None
            evaluator = ImageNetEvaluator(
                             paper_model_name='ResNeXt-101-32x8d',
                             paper_arxiv_id='1611.05431')

            with torch.no_grad():
                for i, (input, target) in enumerate(test_loader):
                    input = input.to(device=device, non_blocking=True)
                    target = target.to(device=device, non_blocking=True)
                    output = model(input)

                    image_ids = [img[0].split('/')[-1].replace('.JPEG', '') for img in test_loader.dataset.imgs[i*test_loader.batch_size:(i+1)*test_loader.batch_size]]

                    evaluator.update(dict(zip(image_ids, list(output.cpu().numpy()))))

            print(evaluator.get_results())

            evaluator.save()
    """

    task = "Image Classification"

    def __init__(self,
                 root: str = '.',
                 paper_model_name: str = None,
                 paper_arxiv_id: str = None,
                 paper_pwc_id: str = None,
                 paper_results: dict = None,
                 pytorch_hub_url: str = None,
                 model_description=None,):
        """Benchmarking function.

        Args:
            root (string): Root directory of the ImageNet Dataset.
            paper_model_name (str, optional): The name of the model from the
                paper - if you want to link your build to a machine learning
                paper. See the ImageNet benchmark page for model names,
                https://www.sotabench.com/benchmark/imagenet, e.g. on the paper
                leaderboard tab.
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

                    {'Top 1 Accuracy': 0.543, 'Top 5 Accuracy': 0.654}.

                Ensure that the metric names match those on the sotabench
                leaderboard - for ImageNet it should be 'Top 1 Accuracy' and
                'Top 5 Accuracy'.
            pytorch_hub_url (str, optional): Optional linking to PyTorch Hub
                url if your model is linked there; e.g:
                'nvidia_deeplearningexamples_waveglow'.
            model_description (str, optional): Optional model description.
        """

        root = self.root = os.path.expanduser(root)

        self.paper_model_name = paper_model_name
        self.paper_arxiv_id = paper_arxiv_id
        self.paper_pwc_id = paper_pwc_id
        self.paper_results = paper_results
        self.pytorch_hub_url = pytorch_hub_url
        self.model_description = model_description

        self.top1 = None
        self.top5 = None

        self.load_targets()

        self.outputs = {}
        self.results = None

    def load_targets(self):

        download_url(
            url=ARCHIVE_DICT['labels']['url'],
            root=self.root,
            md5=ARCHIVE_DICT['labels']['md5'])

        with open(os.path.join(self.root, 'imagenet_val_targets.pkl'), 'rb') as handle:
            self.targets = pickle.load(handle)

    def update(self, output_dict: dict):
        """
        Update the evaluator with new results


        :param output_dict (dict): Where keys are image IDs, and each value should be an 1D np.ndarray of size 1000
        containing logits for that image ID.
        :return: void - updates self.outputs with the new IDSs and prediction

        Examples:
            Update the evaluator with two results:

            .. code-block:: python

                my_evaluator.update({'ILSVRC2012_val_00000293': np.array([1.04243, ...]),
                'ILSVRC2012_val_00000294': np.array([-2.3677, ...])})
        """

        self.outputs = dict(list(self.outputs.items()) + list(output_dict.items()))

    def get_results(self):
        """
        Gets the results for the evaluator. This method only runs if predictions for all 5,000 ImageNet validation
        images are available. Otherwise raises an error and informs you of the missing or unmatched IDs.

        :return: dict with Top 1 and Top 5 Accuracy
        """

        if set(self.targets.keys()) != set(self.outputs.keys()):
            missing_ids = set(self.targets.keys()) - set(self.outputs.keys())
            unmatched_ids = set(self.outputs.keys()) - set(self.targets.keys())

            if len(unmatched_ids) > 0:
                raise AttributeError('''There are {mis_no} missing and {un_no} unmatched image IDs\n\n'''
                                     '''Missing IDs are {missing}\n\n'''
                                     '''Unmatched IDs are {unmatched}'''.format(mis_no=len(missing_ids),
                                                                                un_no=len(unmatched_ids),
                                                                                missing=missing_ids,
                                                                                unmatched=unmatched_ids))
            else:
                raise AttributeError('''There are {mis_no} missing image IDs\n\n'''
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

        return self.results

    def save(self):
        """
        Calculate results and then put into a BenchmarkResult object

        On the sotabench.com server, this will produce a JSON file serialisation and results will be recorded
        on the platform.

        :return: BenchmarkResult object with results and metadata
        """

        if not self.results:
            self.get_results()

        return BenchmarkResult(
            task=self.task,
            config={},
            dataset='ImageNet',
            results=self.results,
            pytorch_hub_id=self.pytorch_hub_url,
            model=self.paper_model_name,
            model_description=self.model_description,
            arxiv_id=self.paper_arxiv_id,
            pwc_id=self.paper_pwc_id,
            paper_results=self.paper_results,
            run_hash=None,
        )
