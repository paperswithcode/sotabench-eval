import numpy as np
from sotabenchapi.core import BenchmarkResult, check_inputs
import tqdm

from sotabencheval.utils import AverageMeter
from .utils import top_k_accuracy_score

class ImageNet:
    task = "Image Classification"

    @classmethod
    @check_inputs
    def benchmark(
        cls,
        results_dict,
        data_root: str = "./.data/vision/imagenet",
        paper_model_name: str = None,
        paper_arxiv_id: str = None,
        paper_pwc_id: str = None,
        paper_results: dict = None,
        pytorch_hub_url: str = None,
        model_description=None,
    ) -> BenchmarkResult:
        """Benchmarking function.

        Args:
            results_dict (dict): dict with keys as image IDs and values as a 1D 1000 x 1 np.ndarrays
            of logits. For example: {'ILSVRC2012_val_00000293': array([1.27443619e+01, ...]), ...}. There
            should be 5000 key/value pairs for the validation set.
            data_root (str): The location of the ImageNet dataset - change this
                parameter when evaluating locally if your ImageNet data is
                located in a different folder (or alternatively if you want to
                download to an alternative location).
            model_description (str, optional): Optional model description.
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
        """

        print("Benchmarking on ImageNet...")

        config = locals()

        try:
            test_dataset = cls.dataset(
                data_root,
                split="val",
                transform=cls.input_transform,
                target_transform=None,
                download=True,
            )
        except Exception:
            test_dataset = cls.dataset(
                data_root,
                split="val",
                transform=cls.input_transform,
                target_transform=None,
                download=False,
            )

        top1 = AverageMeter()
        top5 = AverageMeter()

        for i, (_, target) in enumerate(tqdm.tqdm(test_dataset)):
            image_id = test_dataset.imgs[i][0].split('/')[-1].replace('.JPEG', '')
            output = results_dict[image_id]
            target = target.cpu().numpy()

            prec1 = top_k_accuracy_score(y_true=target, y_pred=np.array([output]), k=1)
            prec5 = top_k_accuracy_score(y_true=target, y_pred=np.array([output]), k=5)
            top1.update(prec1, 1)
            top5.update(prec5, 1)

        final_results = {
            'Top 1 Accuracy': prec1.avg,
            'Top 5 Accuracy': prec5.avg
        }

        print(
            " * Acc@1 {top1:.3f} Acc@5 {top5:.3f}".format(
                top1=final_results["Top 1 Accuracy"],
                top5=final_results["Top 5 Accuracy"],
            )
        )

        return BenchmarkResult(
            task=cls.task,
            config=config,
            dataset=cls.dataset.__name__,
            results=final_results,
            pytorch_hub_id=pytorch_hub_url,
            model=paper_model_name,
            model_description=model_description,
            arxiv_id=paper_arxiv_id,
            pwc_id=paper_pwc_id,
            paper_results=paper_results,
            run_hash=None,
        )
