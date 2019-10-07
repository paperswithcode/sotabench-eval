# Some of the processing logic here is based on the torchvision COCO dataset

import os
from itertools import islice
from enum import Enum
from pathlib import Path

import numpy as np

from sotabenchapi.check import in_check_mode
from sotabenchapi.client import Client
from sotabenchapi.core import BenchmarkResult, check_inputs
from sotabencheval.core import BaseEvaluator
from sotabencheval.utils import calculate_batch_hash, extract_archive, change_root_if_server, is_server

def to_numpy(*args):
    def convert(a):
        if hasattr(a, 'cpu') and hasattr(a, 'numpy'):
            return a.cpu().numpy()
        if isinstance(a, list):
            return np.array(a)
        return a 
    return [convert(a) for a in args]

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
def gether_probs(log_probabilities, targets):
    if hasattr(log_probabilities, 'cpu') and hasattr(log_probabilities, 'numpy'):
        probs = log_probabilities.gather(-1,
                                            targets.unsqueeze(-1))
        self._neglogloss += - probs.sum().cpu().item()
        self._data_set_size += int(targets.numel())
    else: # fall back to numpy implementation that is 4 times slower than pytorch
        vocab_sz = int(log_probabilities.shape[-1])
        log_probabilities, targets =  to_numpy(log_probabilities, targets)
        log_probabilities = log_probabilities.reshape(-1, vocab_sz)
        targets = targets.reshape(-1)
        probs = log_probabilities[np.arange(log_probabilities.shape[0]), targets]
    return probs, targets     
class WikiTextEvaluator(BaseEvaluator):
    task = "Language Modelling"
    dataset = None # defined in subclass

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
        self.dataset.get_path(self.local_root)
        
    def reset(self):
        self._neglogloss = 0
        self._data_set_size = 0 
    
    def add(self, log_probabilities, targets):
        """
            log_probabilietes - 
                float - summed log probability of targets
                [bs x bptt] array of floats - log probability of each target log_proabilites.shape == targets.shape
                [bs x bptt x vocab_size]  - og probability of each word, we will gather correct probablites based on target
        """
        if isinstance(log_probabilities, float):
            log_probabilities = np.array([log_probabilities]) #  for sum to work
        elif log_probabilities.shape[:-1] == targets.shape:
            log_probabilities, targets = gether_probs(log_probabilities, targets)
        else:
            assert log_probabilities.shape == targets.shape, "log_probs have to be ether gethered log probabilities of targets or all probablities" 
        self._neglogloss += - float(log_probabilities.sum())
        self._data_set_size += int(targets.shape[0])

        if not self.first_batch_processed:
            content = self.cache_values(
                probs=to_numpy(log_probabilities)[0].reshape(-1))
            self.batch_hash = calculate_batch_hash(content)
            self.first_batch_processed = True
        return self.results
    
    def print_stats(self):
        print("Perplexity:", np.exp(self._neglogloss / self.dataset.testset_size), 
              "NeglogLoss:", self._neglogloss, "Tokens Count:", self._data_set_size)

    def get_results(self):
        if self.cached_results:
            return self.results
        perplexity = np.exp(self._neglogloss /
                            self.dataset.testset_size)
                            
        self.results = {
            'Test perplexity': perplexity
        }
        return self.results

    def save(self):
        return super().save(dataset=self.dataset.pwc_name)


class WikiText103Evaluator(WikiTextEvaluator):
    dataset = WikiTextDataset.WikiText103

class WikiText2Evaluator(WikiTextEvaluator):
    dataset = WikiTextDataset.WikiText2
