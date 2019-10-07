from sotabencheval.core import BaseEvaluator
from sotabencheval.utils import calculate_batch_hash, change_root_if_server, is_server, get_max_memory_allocated
from sotabencheval.question_answering.utils import *
from typing import Dict
from enum import Enum
from pathlib import Path
import json
import time

class SQuADVersion(Enum):
    V11 = 'v1.1'
    V20 = 'v2.0'


class SQuADEvaluator(BaseEvaluator):
    task = "Question Answering"

    def __init__(self,
                 local_root: str = '.',
                 dataset_filename: str = None,
                 model_name: str = None,
                 paper_arxiv_id: str = None,
                 paper_pwc_id: str = None,
                 paper_results: dict = None,
                 model_description=None,
                 version: SQuADVersion = SQuADVersion.V20):
        super().__init__(model_name, paper_arxiv_id, paper_pwc_id, paper_results, model_description)
        self.root = change_root_if_server(root=local_root,
                                          server_root=".data/nlp/squad")
        self.version = version
        if dataset_filename is None or is_server():
            dataset_filename = "dev-{}.json".format(version.value)
        self.dataset_path = Path(self.root) / dataset_filename

        self.metrics = SQuADMetrics(self.dataset_path, version)

    def add(self, answers: Dict[str, str]):
        self.metrics.add(answers)

        if not self.first_batch_processed and self.metrics.has_data:
            self.batch_hash = calculate_batch_hash(
                self.cache_values(answers=self.metrics.answers,
                                  metrics=self.metrics.get_results(ignore_missing=True))
            )
            self.first_batch_processed = True

    def reset(self):
        self.metrics.reset()

    def get_results(self):
        if self.cached_results:
            return self.results
        self.results = self.metrics.get_results()
        self.speed_mem_metrics['Max Memory Allocated (Total)'] = get_max_memory_allocated()

        return self.results

    def save(self):
        dataset = "SQuAD{} dev".format(self.metrics.version.value[1:])

        if not self.cached_results:
            exec_speed = (time.time() - self.init_time)
            self.speed_mem_metrics['Tasks / Evaluation Time'] = len(self.metrics.answers) / exec_speed
            self.speed_mem_metrics['Tasks'] = len(self.metrics.answers)
            self.speed_mem_metrics['Evaluation Time'] = exec_speed
        else:
            self.speed_mem_metrics['Tasks / Evaluation Time'] = None
            self.speed_mem_metrics['Tasks'] = None
            self.speed_mem_metrics['Evaluation Time'] = None

        return super().save(dataset=dataset)


# todo: aggregate batches so that size of the batch used for caching does not depend on evaluation batch size
CACHE_BATCH_SIZE = 1024


class SQuADMetrics:
    def __init__(self, dataset_path: Path, version: SQuADVersion = SQuADVersion.V20):
        self.version = version
        self.answers = {}
        self._dataset = self._load_dataset(dataset_path)
        self._results = None

    def _load_dataset(self, path):
        with open(path, 'rt') as f:
            ds = json.load(f)
        if 'version' not in ds or 'data' not in ds:
            raise ValueError("Incorrect dataset format, either 'version' or 'data' is missing")
        version = ds['version'].strip().lower()
        if version and version[0] != 'v':
            version = 'v'+version
        if self.version.value != version:
            raise ValueError("Incorrect dataset version, found {} but was expecting {}"
                             .format(version, self.version.value))
        return ds['data']

    def reset(self):
        self._results = None
        self.answers = {}

    def add(self, answers: Dict[str, str]):
        if not answers:
            print("Empty batch added to results")
            return
        if set(self.answers.keys()) & set(answers.keys()):
            print("Multiple predictions for a single question")

        self.answers.update(answers)

    def evaluate(self, ignore_missing=False):
        if ignore_missing:
            dataset = [{'paragraphs': [
                {'qas': [qa for qa in paragraph['qas'] if qa['id'] in self.answers]}
                for paragraph in article['paragraphs']
            ]} for article in self._dataset]
        else:
            dataset = self._dataset
        if self.version == SQuADVersion.V11:
            eval_fn = evaluate_v11
        else:
            eval_fn = evaluate_v20
        results = eval_fn(dataset, self.answers)
        self._results = {
            'EM': results['exact_match'] / 100.0,
            'F1': results['f1'] / 100.0
        }

    @property
    def has_data(self):
        return bool(self.answers)

    def get_results(self, ignore_missing=False):
        self.evaluate(ignore_missing)

        return self._results
