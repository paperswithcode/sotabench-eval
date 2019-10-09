import os
import time
from itertools import islice
from enum import Enum
from pathlib import Path

import numpy as np

from sotabenchapi.check import in_check_mode
from sotabenchapi.client import Client
from sotabenchapi.core import BenchmarkResult, check_inputs
from sotabencheval.core import BaseEvaluator
from sotabencheval.utils import calculate_batch_hash, extract_archive, change_root_if_server, is_server, get_max_memory_allocated

class WikiTextDataset(Enum):
    WikiText103 = ('WikiText-103', 245569, 267735)
    WikiText2 = ('WikiText-2', 245569, 33278)
    
    def __init__(self, pwc_name, testset_size, vocab_size):
        self.pwc_name = pwc_name
        self.testset_size = testset_size
        self.vocab_size = vocab_size
    
    def get_path(self, local_root, local_unzip=False):
        root = Path(change_root_if_server(root=local_root,
                                          server_root=".data/nlp/" + self.pwc_name.lower()))
        zip_name = self.pwc_name.lower() + "-v1.zip"
        dataset_path = root / "wiki.test.tokens"
        if not dataset_path.exists(): # unzip
            extract_archive(str(root / zip_name), to_path=root.parent)
        return dataset_path

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
        log_probs - `torch.tensor`/`np.ndarray` shape [bs x seq_len x vocab_sz] 
                     with model activations after `log_softmax`, with log probability of each word in the vocab
        targets - `torch.tensor`/`np.ndarray` shape [bs x seq_len] with ground truth words
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
    dataset = None # defined in a subclass

    def __init__(self,
                 local_root: str = '.',
                 model_name: str = None,
                 paper_arxiv_id: str = None,
                 paper_pwc_id: str = None,
                 paper_results: dict = None,
                 model_description=None,
                 text_transformation: bool = False,
                 subword_tokenization: bool = False,
                 dataset=None):

        super().__init__(model_name, paper_arxiv_id,
                         paper_pwc_id, paper_results, model_description)
        if dataset is not None:
            self.dataset = dataset
        self.subword_tokenization = subword_tokenization
        self.text_transformation = text_transformation
        self.local_root = local_root 
        self.reset()
    
    @property
    def dataset_path(self):
        return self.dataset.get_path(self.local_root)
        
    def reset(self):
        self._neglogloss = 0
        self._data_set_size = 0 
    
    def add(self, log_probs, targets):
        """
            log_probs - 
                float - summed log probability of targets
                [bs x seq_len] array of floats - log probability of each target log_probs.shape == targets.shape
                [bs x seq_len x vocab_size]  - og probability of each word, we will gather correct probabilites based on target
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
        super().print_results()
        print("Perplexity:", np.exp(self._neglogloss / self.dataset.testset_size), 
              "NeglogLoss:", self._neglogloss, "Tokens Count:", self._data_set_size)
    
    print_stats = print_results
    
    def get_results(self):
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
        return super().save(dataset=self.dataset.pwc_name)


class WikiText103Evaluator(WikiTextEvaluator):
    dataset = WikiTextDataset.WikiText103

class WikiText2Evaluator(WikiTextEvaluator):
    dataset = WikiTextDataset.WikiText2
