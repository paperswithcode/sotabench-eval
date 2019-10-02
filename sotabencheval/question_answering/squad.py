from sotabencheval.core import Submission
from sotabencheval.utils import calculate_batch_hash, change_root_if_server, is_server
from sotabencheval.question_answering.utils import *
from typing import Dict
from enum import Enum
from pathlib import Path
import json


class SQuADVersion(Enum):
    V11 = 'v1.1'
    V20 = 'v2.0'


class SQuADSubmission(Submission):
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
        if dataset_filename is None or is_server():
            dataset_filename = "dev-{}.json".format(version.value)
        self.dataset_path = Path(self.root) / dataset_filename

        self.evaluator = SQuADEvaluator(self.dataset_path, version)

    def add(self, answers: Dict[str, str]):
        self.evaluator.add(answers)

        if not self.first_batch_processed and self.evaluator.has_data:
            self.batch_hash = calculate_batch_hash(
                self.cache_values(answers=self.evaluator.answers,
                                  metrics=self.evaluator.get_results(ignore_missing=True))
            )
            self.first_batch_processed = True

    def get_results(self):
        if self.cached_results:
            return self.results
        self.results = self.evaluator.get_results()
        return self.results

    def save(self):
        dataset = "SQuAD {}".format(self.evaluator.version.value)
        return super().save(dataset=dataset)


# todo: aggregate batches so that size of the batch used for caching does not depend on evaluation batch size
CACHE_BATCH_SIZE = 1024


class SQuADEvaluator:
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
