from sotabencheval.core import BaseEvaluator
from sotabencheval.utils import calculate_batch_hash, change_root_if_server, is_server
from sotabencheval.machine_translation.languages import Language
from sotabencheval.machine_translation.metrics import TranslationMetrics
from typing import Dict, List
from pathlib import Path


class WMTEvaluator(BaseEvaluator):
    task = "Machine Translation"

    def __init__(self,
                 source_lang: Language,
                 target_lang: Language,
                 local_root: str = '.',
                 source_dataset_filename: str = None,
                 target_dataset_filename: str = None,
                 model_name: str = None,
                 paper_arxiv_id: str = None,
                 paper_pwc_id: str = None,
                 paper_results: dict = None,
                 model_description=None):
        super().__init__(model_name, paper_arxiv_id, paper_pwc_id, paper_results, model_description)
        self.root = change_root_if_server(root=local_root,
                                          server_root=".data/nlp/wmt")
        self.source_lang = source_lang
        self.target_lang = target_lang

        if source_dataset_filename is None or is_server():
            source_dataset_filename = "newstest2019-{0}{1}-src.{0}.sgm". \
                format(self.source_lang.value, self.target_lang.value)

        if target_dataset_filename is None or is_server():
            target_dataset_filename = "newstest2019-{0}{1}-ref.{1}.sgm". \
                format(self.source_lang.value, self.target_lang.value)

        self.source_dataset_path = Path(self.root) / source_dataset_filename
        self.target_dataset_path = Path(self.root) / target_dataset_filename

        self.metrics = TranslationMetrics(self.source_dataset_path, self.target_dataset_path)

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
        return self.results

    def save(self):
        datasets = {
            ("2019", Language.English, Language.German): "WMT2019 English German"
        }
        dataset = datasets.get(("2019", Language.English, Language.German))
        return super().save(dataset=dataset)

