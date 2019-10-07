from dataclasses import dataclass
from bs4 import BeautifulSoup
from pathlib import Path
from typing import Dict, List
from collections import OrderedDict
from sacrebleu import corpus_bleu


MIN_CACHE_BATCH_SIZE = 32


class TranslationMetrics:
    def __init__(self, source_dataset_path: Path, target_dataset_path):
        self._src_dataset_path = source_dataset_path
        self._dst_dataset_path = target_dataset_path
        self.answers = {}
        self.source_documents, self.source_segments = self._load_dataset(self._src_dataset_path)
        self._target_documents, self._target_segments = self._load_dataset(self._dst_dataset_path)
        self._results = None

    def _load_dataset(self, dataset_path):
        documents = read_sgm_file(dataset_path)
        segments = OrderedDict([(segment.id, segment.text) for doc in documents for segment in doc.segments])
        return documents, segments

    def add(self, answers: Dict[str, str]):
        if not answers:
            print("Empty batch added to results")
            return
        if set(self.answers.keys()) & set(answers.keys()):
            print("Multiple translations for the same segment")
        self.answers.update(answers)

    def reset(self):
        self._results = None
        self.answers = {}

    def evaluate(self, ignore_missing=False):
        if ignore_missing:
            keep = set(self.answers.keys())
            target_segments = {sid: text for sid, text in self._target_segments.items() if sid in keep}
        else:
            target_segments = self._target_segments
        references = [[target for target in target_segments.values()]]
        answers = [self.answers.get(sid, "") for sid in target_segments]
        bleu = corpus_bleu(answers, references)
        self._results = {
            'BLEU score': bleu.score
        }

    @property
    def has_data(self):
        return len(self.answers) >= MIN_CACHE_BATCH_SIZE

    def get_results(self, ignore_missing=False):
        self.evaluate(ignore_missing)
        return self._results


@dataclass
class Segment:
    id: str
    text: str


@dataclass
class Document:
    id: str
    segments: List[Segment]


def read_sgm_file(path):
    with open(path, 'rb') as f:
        soup = BeautifulSoup(f.read(), features="html.parser")

    return [
        Document(
            id=doc['docid'],
            segments=[
                Segment(
                    id=doc['docid'] + '#' + seg['id'],
                    text=seg.text
                ) for seg in doc.find_all('seg')
            ]
        ) for doc in soup.find_all('doc')
    ]
