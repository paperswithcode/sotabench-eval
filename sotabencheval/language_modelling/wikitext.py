import time
from enum import Enum
from pathlib import Path

import numpy as np

from sotabencheval.core import BaseEvaluator
from sotabencheval.utils import calculate_batch_hash, extract_archive, change_root_if_server, is_server, get_max_memory_allocated


class WikiTextDataset(Enum):
    """Enum used to select the dataset on which evaluation is executed. """
    WikiText103 = ('WikiText-103', 245569, 267735)
    WikiText2 = ('WikiText-2', 245569, 33278)
    
    def __init__(self, pwc_name, testset_size, vocab_size):
        """
        Creates an enum instance
        :param pwc_name: the name of the dataset as it is found on paperswithcode leaderboard
        :param testset_size: the size of the test set in words
        :param vocab_size: the size of the dataset vocabluary
        """
        self.pwc_name = pwc_name
        self.testset_size = testset_size
        self.vocab_size = vocab_size
    
    def _get_path(self, local_root, local_unzip=False):
        root = Path(change_root_if_server(root=local_root,
                                          server_root=".data/nlp/" + self.pwc_name.lower()))
        zip_name = self.pwc_name.lower() + "-v1.zip"
        dataset_path = root / "wiki.test.tokens"
        if not dataset_path.exists(): # unzip
            extract_archive(str(root / zip_name), to_path=root.parent)
        return dataset_path
    
    get_path = _get_path # deprecated API, for backward compatibility with existing benchmarks
    
    def get_test_set_path(self, local_root):
        """ 
        Unzips the datasets and returns path to "wiki.test.tokens" 
        :param local_root: Path to the directory where the dataset files are located locally.
            Ignored when run on sotabench server.
        """
        return self.get_path(local_root).parent / "wiki.test.tokens"

    def get_validation_set_path(self, local_root):
        """ 
        Unzips the datasets and returns path to "wiki.test.tokens" 
        :param local_root: Path to the directory where the dataset files are located locally.
            Ignored when run on sotabench server.
        """
        return self.get_path(local_root).parent / "wiki.valid.tokens"

def _to_numpy(*args):
    def convert(a):
        if hasattr(a, 'cpu') and hasattr(a, 'numpy'):
            return a.cpu().numpy()
        if isinstance(a, list):
            return np.array(a)
        return a
    return [convert(a) for a in args]

def _gather_probs(log_probs, targets):
    """
    Gather probabilities of each target token, from the model activations after log_softmax
    :param log_probs: - `torch.tensor`/`np.ndarray` shape [bs x seq_len x vocab_sz] 
                         with model activations after `log_softmax`, with log probability of each word in the vocab
    :param targets: - `torch.tensor`/`np.ndarray` shape [bs x seq_len] with ground truth words
    """
    if hasattr(log_probs, 'gather'):
        # if we work with torch this method is faster than numpy implementation
        probs = log_probs.gather(-1, targets.unsqueeze(-1))
    elif isinstance(log_probs, np.ndarray):
        # use slower numpy implementation if we have ndarrays
        vocab_sz = int(log_probs.shape[-1])
        log_probs, targets =  _to_numpy(log_probs, targets)
        log_probs = log_probs.reshape(-1, vocab_sz)
        targets = targets.reshape(-1)
        probs = log_probs[np.arange(log_probs.shape[0]), targets]
    return _to_numpy(probs, targets)   

    
class WikiTextEvaluator(BaseEvaluator):
    task = "Language Modelling"
    dataset = None  # defined in a subclass

    def __init__(self,
                 local_root: str = '.',
                 model_name: str = None,
                 paper_arxiv_id: str = None,
                 paper_pwc_id: str = None,
                 paper_results: dict = None,
                 model_description=None,                 
                 subword_tokenization: bool = False,
                 text_transformation: bool = False,
                 dataset=None):
        """
        Creates an evaluator for one of the WikiText benchmarks.

        :param local_root: Path to the directory where the dataset files are located locally.
            Ignored when run on sotabench server.
        :param model_name: The name of the model from the
            paper - if you want to link your build to a model from a
            machine learning paper. See the WikiText-103 benchmarks page for model names,
            (f.e., https://sotabench.com/benchmarks/language-modelling-on-wikitext-103)
            on the paper leaderboard or models yet to try tab.
        :param paper_arxiv_id: Optional linking to arXiv if you
            want to link to papers on the leaderboard; put in the
            corresponding paper's arXiv ID, e.g. '1901.02860'.
        :param paper_pwc_id: Optional linking to Papers With Code;
            put in the corresponding papers with code URL slug, e.g.
            "transformer-xl-attentive-language-models"
        :param paper_results: If the paper model you are reproducing
            does not have model results on sotabench.com, you can specify
            the paper results yourself through this argument, where keys
            are metric names, values are metric values. e.g:

                    {'Test perplexity': 18.2 }.

            Ensure that the metric names match those on the sotabench
            leaderboard - for WikiText benchmarks it should be `Test perplexity`.
        :param model_description: Optional model description.
        :param subword_tokenization: Should be set to `True` if your model use subword tokens defaults to `False`,  
        :param text_transformation: Should be set to  `True` if you use detokenizers that removes moses artefacts, f.e. in zero shoot setting,  
        :param dataset: internal paramtere do not set in subclasses.
        """
        super().__init__(model_name, paper_arxiv_id,
                         paper_pwc_id, paper_results, model_description)
        if dataset is not None:
            self.dataset = dataset
        self.subword_tokenization = subword_tokenization
        self.text_transformation = text_transformation
        self.local_root = local_root 
        self._neglogloss = 0
        self._data_set_size = 0 
    
    @property
    def dataset_path(self): # deprecated 
        return self.dataset.get_path(self.local_root)

    @property
    def test_set_path(self):
        """Returns path to test set, uses `self.local_root` when it is not on the server"""
        return self.get_test_set_path(self.local_root)

    @classmethod
    def get_test_set_path(cls, local_root):
        """
        Unzips the datasets and returns path to "wiki.test.tokens" 
        :param local_root: Path to the directory where the dataset files are located locally.
            Ignored when run on sotabench server.
        """
        return cls.dataset.get_test_set_path(local_root)
        
    def reset(self):
        """
        Removes already added results


        When checking if the model should be rerun on whole dataset it is first run on a smaller subset
        and the results are compared with values cached on sotabench server (the check is not performed
        when running locally.) Ideally, the smaller subset is just the first batch, so no additional
        computation is needed. However, for more complex multistage pipelines it maybe simpler to
        run a model twice - on a small dataset and (if necessary) on the full dataset. In that case
        :func:`reset` needs to be called before the second run so values from the first run are not reported.

        .. seealso:: :func:`cache_exists`
        .. seealso:: :func:`reset_time`
        """
        self._neglogloss = 0
        self._data_set_size = 0 
    
    def add(self, log_probs, targets):
        """
        Updates the evaluator with new results

        :param log_probs: `np.ndarray` or `torch.tensor` with log probability of target tokens can be either:
            - a 0d tensor
                summed log probability of all `targets` tokens, or 
            - a 2d tensor [bs x seq_len]
                log probabilities of each target token, the shape of `log_probs`, `targets` must match.
            - a 3d tensor [bs x seq_len x vocab_size] 
                 distribution of log probabilities for each position in the sequence,
                 we will gather the probabilities of target tokens for you.
        :param targets: a `np.ndarray` or `torch.tensor`  with ids of ground truth tokens.

        Examples:
            Update the evaluator with a result for a sentence with 10 tokens:

            .. code-block:: python
                log_probs = np.array([[ 32, 582, 2731, 19, 1, 786,  5, 98693, 55362, 5 ]])
                targets = np.array([[ -9.8461,  -9.3343, -17.8042, -11.2006, -22.3345, -14.4665,  -2.0055,
                                    -14.2044, -14.7545,  -5.7888]])
                my_evaluator.add(log_probs, targets)
        """
        if isinstance(log_probs, float):
            log_probs = np.array([log_probs]) #  for sum to work
        elif log_probs.shape[:-1] == targets.shape:
            log_probs, targets = _gather_probs(log_probs, targets)
        else:
            assert log_probs.shape == targets.shape, f"log_probs have to be ether gathered log probabilities of targets or all probabilites, received {log_probs.shape} {repr(log_probs)}"
        self._neglogloss += - float(log_probs.sum())
        self._data_set_size += int(np.prod(list(targets.shape)))

        if not self.first_batch_processed:
            content = self.cache_values(
                probs=_to_numpy(log_probs)[0].reshape(-1),
                api_version=3)
            self.batch_hash = calculate_batch_hash(content)
            self.first_batch_processed = True
        return self.results
    
    def print_results(self):
        """ Calculates and print results. """
        super().print_results()
        print("Perplexity:", np.exp(self._neglogloss / self.dataset.testset_size), 
              "NeglogLoss:", self._neglogloss, "Tokens Count:", self._data_set_size)
    
    print_stats = print_results
    
    def get_results(self):
        """ 
        Calculates the perplexity and measure the performance of the model
        
        :return: dict with `Test perplexity`
        """
        if self.cached_results:
            return self.results
        perplexity = np.exp(self._neglogloss /
                            self.dataset.testset_size)
                            
        self.results = {
            'Test perplexity': perplexity
        }
        self.speed_mem_metrics['Max Memory Allocated (Total)'] = get_max_memory_allocated()
        exec_speed = (time.time() - self.init_time)
        count = self.dataset.testset_size
        self.speed_mem_metrics['Tasks / Evaluation Time'] = count / exec_speed
        self.speed_mem_metrics['Tasks'] = count
        self.speed_mem_metrics['Evaluation Time'] = exec_speed
        return self.results

    def save(self):
        """Save results to the server databese/"""
        return super().save(dataset=self.dataset.pwc_name)


class WikiText103Evaluator(WikiTextEvaluator):
    """`WikiText103 <https://sotabench.com/benchmarks/language-modelling-on-wikitext-103>`_ benchmark.

    Examples:
        Evaluate a language model from the transformers repository:

        .. code-block:: python

            import torch
            from tqdm import tqdm
            from sotabencheval.language_modelling import WikiText103Evaluator

            model = torch.hub.load('huggingface/transformers', 'modelWithLMHead', 'transfo-xl-wt103').to("cuda")
            tokenizer = torch.hub.load('huggingface/transformers', 'tokenizer', 'transfo-xl-wt103')

            evaluator = WikiText103Evaluator(
                model_name="Transformer-XL Large", 
                paper_arxiv_id="1901.02860",
                paper_pwc_id="transformer-xl-attentive-language-models",
                local_root='/content/wikitext-103'
            )

            with evaluator.test_set_path.open() as f:
                test_data = torch.tensor(tokenizer.encode(f.read()))

            seq_len = 128
            with torch.no_grad():
                evaluator.reset_timer()
                model.eval()
                X, Y, mems = test_data[None, :-1], test_data[None, 1:], None
                for s in tqdm(range(0, X.shape[-1], seq_len)):
                    x,y = X[..., s:s+seq_len].to("cuda"), Y[..., s:s+seq_len].to("cuda")
                    log_probs, mems, *_ = model(input_ids=x, mems=mems)
                    evaluator.add(log_probs, y)
                    if evaluator.cache_exists:
                        break
            evaluator.save()
            evaluator.print_results()
    """
    dataset = WikiTextDataset.WikiText103


class WikiText2Evaluator(WikiTextEvaluator):
    """`WikiText103 <https://sotabench.com/benchmarks/language-modelling-on-wikitext-2>`_ benchmark.

    Examples:
        Evaluate a language model from the transformers repository:

        .. code-block:: python

            import torch
            from tqdm import tqdm
            from sotabencheval.language_modelling import WikiText2Evaluator

            model = torch.hub.load('huggingface/transformers', 'modelWithLMHead', 'transfo-xl-wt103').to("cuda")
            tokenizer = torch.hub.load('huggingface/transformers', 'tokenizer', 'transfo-xl-wt103')

            evaluator = WikiText2Evaluator(
                model_name="Transformer-XL Large", 
                paper_arxiv_id="1901.02860",
                paper_pwc_id="transformer-xl-attentive-language-models",
                local_root='/content/wikitext-2'
            )

            with evaluator.test_set_path.open() as f:
                test_data = torch.tensor(tokenizer.encode(f.read()))

            seq_len = 128
            with torch.no_grad():
                evaluator.reset_timer()
                model.eval()
                X, Y, mems = test_data[None, :-1], test_data[None, 1:], None
                for s in tqdm(range(0, X.shape[-1], seq_len)):
                    x,y = X[..., s:s+seq_len].to("cuda"), Y[..., s:s+seq_len].to("cuda")
                    log_probs, mems, *_ = model(input_ids=x, mems=mems)
                    evaluator.add(log_probs, y)
                    if evaluator.cache_exists:
                        break
            evaluator.save()
            evaluator.print_results()
    """
    dataset = WikiTextDataset.WikiText2
