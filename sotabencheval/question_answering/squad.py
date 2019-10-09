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
    """Evaluator for Stanford Question Answering Dataset v1.1 and v2.0 benchmarks.

    Examples:
        Evaluate a BiDAF model from the AllenNLP repository on SQuAD 1.1 development set:

        .. code-block:: python

            from sotabencheval.question_answering import SQuADEvaluator, SQuADVersion

            from allennlp.data import DatasetReader
            from allennlp.data.iterators import DataIterator
            from allennlp.models.archival import load_archive
            from allennlp.nn.util import move_to_device

            def load_model(url, batch_size=64):
                archive = load_archive(url, cuda_device=0)
                model = archive.model
                reader = DatasetReader.from_params(archive.config["dataset_reader"])
                iterator_params = archive.config["iterator"]
                iterator_params["batch_size"] = batch_size
                data_iterator = DataIterator.from_params(iterator_params)
                data_iterator.index_with(model.vocab)
                return model, reader, data_iterator

            def evaluate(model, dataset, data_iterator, evaluator):
                model.eval()
                evaluator.reset_time()
                for batch in data_iterator(dataset, num_epochs=1, shuffle=False):
                    batch = move_to_device(batch, 0)
                    predictions = model(**batch)
                    answers = {metadata['id']: prediction
                               for metadata, prediction in zip(batch['metadata'], predictions['best_span_str'])}
                    evaluator.add(answers)
                    if evaluator.cache_exists:
                        break

            evaluator = SQuADEvaluator(local_root="data/nlp/squad", model_name="BiDAF (single)",
                paper_arxiv_id="1611.01603", version=SQuADVersion.V11)

            model, reader, data_iter =\
                load_model("https://allennlp.s3.amazonaws.com/models/bidaf-model-2017.09.15-charpad.tar.gz")
            dataset = reader.read(evaluator.dataset_path)
            evaluate(model, dataset, data_iter, evaluator)
            evaluator.save()
            print(evaluator.results)
    """

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
        """
        Creates an evaluator for SQuAD v1.1 or v2.0 Question Answering benchmarks.

        :param local_root: Path to the directory where the dataset files are located locally.
            Ignored when run on sotabench server.
        :param dataset_filename: Local filename of the JSON file with the SQuAD dataset.
            If None, the standard filename is used, based on :param:`version`.
            Ignored when run on sotabench server.
        :param model_name: The name of the model from the
            paper - if you want to link your build to a model from a
            machine learning paper. See the SQuAD benchmarks pages for model names,
            (f.e., https://sotabench.com/benchmarks/question-answering-on-squad11-dev)
            on the paper leaderboard or models yet to try tabs.
        :param paper_arxiv_id: Optional linking to arXiv if you
            want to link to papers on the leaderboard; put in the
            corresponding paper's arXiv ID, e.g. '1907.10529'.
        :param paper_pwc_id: Optional linking to Papers With Code;
            put in the corresponding papers with code URL slug, e.g.
            'spanbert-improving-pre-training-by'
        :param paper_results: If the paper model you are reproducing
            does not have model results on sotabench.com, you can specify
            the paper results yourself through this argument, where keys
            are metric names, values are metric values. e.g:

                    {'EM': 0.858, 'F1': 0.873}.

            Ensure that the metric names match those on the sotabench
            leaderboard - for SQuAD benchmarks it should be `EM` for exact match
            and `F1` for F1 score. Make sure to use results of evaluation on a development set.
        :param model_description: Optional model description.
        :param version: Which dataset to evaluate on, either `SQuADVersion.V11` or `SQuADVersion.V20`.
        """
        super().__init__(model_name, paper_arxiv_id, paper_pwc_id, paper_results, model_description)
        self.root = change_root_if_server(root=local_root,
                                          server_root=".data/nlp/squad")
        self.version = version
        if dataset_filename is None or is_server():
            dataset_filename = "dev-{}.json".format(version.value)
        self.dataset_path = Path(self.root) / dataset_filename

        self.metrics = SQuADMetrics(self.dataset_path, version)

    def add(self, answers: Dict[str, str]):
        """
        Updates the evaluator with new results

        :param answers: a dictionary, where keys are question ids and values are text answers.
            For unanswerable questions (SQuAD v2.0) the answer should be an empty string.

        Examples:
            Update the evaluator with two results:

            .. code-block:: python

                my_evaluator.add({
                    "57296d571d04691400779413": "itself",
                    "5a89117e19b91f001a626f2d": ""
                })
        """

        self.metrics.add(answers)

        if not self.first_batch_processed and self.metrics.has_data:
            self.batch_hash = calculate_batch_hash(
                self.cache_values(answers=self.metrics.answers,
                                  metrics=self.metrics.get_results(ignore_missing=True))
            )
            self.first_batch_processed = True

    def reset(self):
        """
        Removes already added answers

        When checking if the model should be rerun on whole dataset it is first run on a smaller subset
        and the results are compared with values cached on sotabench server (the check is not performed
        when running locally.) Ideally, the smaller subset is just the first batch, so no additional
        computation is needed. However, for more complex multistage pipelines it may be simpler to
        run the model twice - on a small dataset and (if necessary) on the full dataset. In that case
        :func:`reset` needs to be called before the second run so values from the first run are not reported.

        .. seealso:: :func:`cache_exists`
        .. seealso:: :func:`reset_time`
        """

        self.metrics.reset()
        self.reset_time()

    def get_results(self):
        """
        Gets the results for the evaluator.

        :return: dict with `EM` (exact match score) and `F1`.
        """

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
