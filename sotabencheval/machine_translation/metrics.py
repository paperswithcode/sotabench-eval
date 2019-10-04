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
        self.source_documents, self.source_sentences = self._load_dataset(self._src_dataset_path)
        self._target_documents, self._target_sentences = self._load_dataset(self._dst_dataset_path)
        self._results = None

    def _load_dataset(self, dataset_path):
        documents = read_sgm_file(dataset_path)
        sentences = OrderedDict([(sentence.id, sentence.text) for doc in documents for sentence in doc.sentences])
        return documents, sentences

    def add(self, answers: Dict[str, str]):
        if not answers:
            print("Empty batch added to results")
            return
        if set(self.answers.keys()) & set(answers.keys()):
            print("Multiple translations for the same sentence")
        self.answers.update(answers)

    def reset(self):
        self._results = None
        self.answers = {}

    def evaluate(self, ignore_missing=False):
        if ignore_missing:
            keep = set(self.answers.keys())
            target_sentences = {sid: text for sid, text in self._target_sentences.items() if sid in keep}
        else:
            target_sentences = self._target_sentences
        references = [[target for target in target_sentences.values()]]
        answers = [self.answers.get(sid, "") for sid in target_sentences]
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
class Sentence:
    id: str
    text: str


@dataclass
class Document:
    id: str
    sentences: List[Sentence]


def read_sgm_file(path):
    with open(path, 'rb') as f:
        soup = BeautifulSoup(f.read(), features="html.parser")

    return [
        Document(
            id=doc['docid'],
            sentences=[
                Sentence(
                    id=doc['docid'] + '#' + seg['id'],
                    text=seg.text
                ) for seg in doc.find_all('seg')
            ]
        ) for doc in soup.find_all('doc')
    ]
