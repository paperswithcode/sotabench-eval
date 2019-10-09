import csv
import time

from itertools import zip_longest
from pathlib import Path

from sotabencheval.core import BaseEvaluator
from sotabencheval.utils import calculate_batch_hash, extract_archive, change_root_if_server, is_server, get_max_memory_allocated


def read_csv(path):
    with path.open('r') as f:
        yield from csv.DictReader(f, delimiter='\t')


def get_path(local_root, local_unzip=False):
    root = Path(change_root_if_server(root=local_root,
                                      server_root=".data/nlp/multinli"))
    zip_name = "MNLI.zip"
    dataset_path=root / "MNLI" / "dev_matched.tsv"
    if not dataset_path.exists():  # unzip
        extract_archive(str(root / zip_name), to_path=root)
    return (dataset_path, dataset_path.parent / "dev_mismatched.tsv")


class ClassificationEvaluator:
    def __init__(self, file_path):
        self.dataset_path = file_path
        dataset = list(read_csv(file_path))
        self.targets = {d['pairID']: d['gold_label'] for d in dataset}
        self.dataset = {d['pairID']: (d['sentence1'], d['sentence2']) for d in dataset}
        self.reset()

    def reset(self):    
        self.answers = {}
    
    @property
    def count(self):
        return len(self.answers)
    
    def add(self, pairIds, preds):
        for pairId, pred in zip(pairIds,preds):
            if pairId not in self.targets:
                continue
            if pairId not in self.answers:
                self.answers[pairId] = pred
            else:
                print(f"Double prediction for {pairId} former: {self.answers[pairId]} new: {pred}")
   
    @property
    def has_enough_for_cache_hash(self):
        return self.count >= 100

    @property
    def accuracy(self):
        correct = [self.targets[k] == a for k,a in self.answers.items() if a is not None]
        accuracy = sum(correct) / self.count if self.count > 0 else 0
        if self.count != len(self.targets):
            return (accuracy, f"partial on {self.count} out of {len(self.targets)}")
        return accuracy


class MultiNLI(BaseEvaluator):
    task = "Natural Language Inference"
    dataset = 'MultiNLI'  # defined in subclass

    def __init__(self,
                 local_root: str = '.',
                 model_name: str = None,
                 paper_arxiv_id: str = None,
                 paper_pwc_id: str = None,
                 paper_results: dict = None,
                 model_description=None):

        super().__init__(model_name, paper_arxiv_id,
                         paper_pwc_id, paper_results, model_description)
        self.local_root = local_root
        paths = self.dataset_paths
        self.matched = ClassificationEvaluator(paths[0])
        self.mismatched = ClassificationEvaluator(paths[1])
        self.reset()

    @property
    def dataset_paths(self):
        return get_path(self.local_root)
    
    @property
    def data_generator(self):
        for v1, v2 in zip_longest(self.matched.dataset.items(), self.mismatched.dataset.items()):
            if v1 is not None:
                yield v1
            if v2 is not None:
                yield v2

    def reset(self):
        self.matched.reset()
        self.mismatched.reset()
        self.batch_hash = None
        self.reset_time()

    def add(self, pairIds, predictions):
        """
            pairIDToLabel - Dictionary mapping pairID (str) to label (str)              
        """
        if isinstance(pairIds, str):
            pairIds = [pairIds]
            predictions = [predictions]
        
        self.matched.add(pairIds, predictions)
        self.mismatched.add(pairIds, predictions)
        if self.batch_hash is None and self.matched.count + self.mismatched.count > 100:
            content = self.cache_values(matched=self.matched.answers, mismatched=self.mismatched.answers)
            self.batch_hash = calculate_batch_hash(content)
            self.first_batch_processed = True #TODO: do we need this if we have self.batch_hash


    def get_results(self):
        if self.cached_results:
            return self.results
        self.results = {
            'Matched': self.matched.accuracy,
            'Mismatched': self.mismatched.accuracy
        }
        self.speed_mem_metrics['Max Memory Allocated (Total)'] = get_max_memory_allocated()
        exec_speed = (time.time() - self.init_time)
        count = self.mismatched.count + self.matched.count
        self.speed_mem_metrics['Tasks / Evaluation Time'] = count / exec_speed
        self.speed_mem_metrics['Tasks'] = count
        self.speed_mem_metrics['Evaluation Time'] = exec_speed
        return self.results

    def save(self):

        
        return super().save(dataset=self.dataset)
