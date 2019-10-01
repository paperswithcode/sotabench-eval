# Some of the processing logic here is based on the torchvision COCO dataset

import os
from typing import Generator, Tuple

from sotabenchapi.check import in_check_mode
from sotabenchapi.client import Client
from sotabenchapi.core import BenchmarkResult, check_inputs
from sotabencheval.utils import calculate_batch_hash, extract_archive, change_root_if_server

import numpy as np
from itertools import islice

import dataclasses

import torch # TODO fix it so that we can work with Tensorflow as well.

TASK="language modeling"
def perplexity_evauluate(results_generator: Generator[Tuple[torch.Tensor, torch.Tensor], None , None], limit: int=None):
    set_sz = 0
    neglogloss = 0
    for log_probs, labels in islice(results_generator, limit):
        neglogloss += -log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(1).sum().cpu().item()
        set_sz += int(labels.numel())
    return np.exp(neglogloss / set_sz), set_sz, neglogloss/set_sz, neglogloss

#%%
def map_eval_to_benchmark(dictionary):
    key_map = {
        'model_name':'model',
        'model_description':'model_description',
        'paper_arxiv_id':'arxiv_id',
        'paper_pwc_id':'pwc_id',
        'paper_results':'paper_results',
        'run_hash':'run_hash'
    }# TODO make it noop by changing the BenchmarkResults
    return {key_map[key]: val for key,val in dictionary.items() if key in key_map}


def to_numpy(*args):
    def convert(a):
        if hasattr(a, 'cpu') and hasattr(a, 'numpy'):
            return a.cpu().numpy()
        if isinstance(a, list):
            return np.array(a)
        return a 
    return [convert(a) for a in args]

#TODO: python 3.6 does not have dataclasses !
@dataclasses.dataclass
class WikiText103Eval:
    model_name: str = None
    paper_arxiv_id: str = None
    paper_pwc_id: str = None
    paper_results: dict = None
    model_description: str = None
    text_transformation: bool = False
    subword_tokenization: bool = False
    def __post_init__(self):            
        self.expected_test_data_set_size = 245569            
        self.dataset = 'WikiText-103'
        self.task = TASK
        self.reset()

    def reset(self):
        self._neglogloss = 0
        self._data_set_size = 0 
    
    #TODO handle both tensorflow and pytorch
    def add(self, log_probabilities, labels):
        if hasattr(log_probabilities, 'cpu') and hasattr(log_probabilities, 'numpy'):
            self._neglogloss += -log_probabilities.gather(-1, labels.unsqueeze(-1)).squeeze(1).sum().cpu().item()
            self._data_set_size += int(labels.numel())
        else: # fall back to numpy implementation that is 4 times slower than pytorch
            log_probabilities, labels =  to_numpy(log_probabilities, labels)
            vocab_sz = log_probabilities.shape[-1]
            log_probabilities = log_probabilities.reshape(-1, vocab_sz)
            labels = labels.reshape(-1)
            self._neglogloss += - log_probabilities[np.arange(log_probabilities.shape[0]), labels].sum()
            self._data_set_size += int(labels.shape[0])
        return self.results

    def eval(self, results_generator):
        self.reset()
        for log_probs, labels in results_generator:
            self.add(log_probs, labels)
            if self.check_first_batch():
                return self.save()
        return self.save()

    def check_results(self):
        pass
    
    def check_first_batch(self):
        pass

    @property
    def results(self):
        return dict(
            neglogloss=self._neglogloss,
            tokens_seen=self._data_set_size,
            perplexity=np.exp(self._neglogloss /
                              self.expected_test_data_set_size)
        )
    
    def save(self):
        """
        Calculate results and then put into a BenchmarkResult object

        On the sotabench.com server, this will produce a JSON file serialisation and results will be recorded
        on the platform.

        :return: BenchmarkResult object with results and metadata
        """
        self.check_results()
        description = map_eval_to_benchmark(dataclasses.asdict(self))
        return BenchmarkResult(
            task=self.task,
            dataset=self.dataset,
            config={},
            results=self.results,
            **description
        )
