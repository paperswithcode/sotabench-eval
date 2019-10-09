from sotabencheval.core import BaseEvaluator
from sotabencheval.utils import calculate_batch_hash, change_root_if_server, is_server
from sotabencheval.machine_translation.languages import Language
from sotabencheval.machine_translation.metrics import TranslationMetrics
from sotabencheval.utils import get_max_memory_allocated
from typing import Dict, Callable
from pathlib import Path
from enum import Enum
import time


class WMTDataset(Enum):
    News2014 = "newstest2014"
    News2019 = "newstest2019"


class WMTEvaluator(BaseEvaluator):
    """Evaluator for WMT Machine Translation benchmarks.

    Examples:
        Evaluate a Transformer model from the fairseq repository on WMT2019 news test set:

        .. code-block:: python

            from sotabencheval.machine_translation import WMTEvaluator, WMTDataset, Language
            from tqdm import tqdm
            import torch

            evaluator = WMTEvaluator(
                dataset=WMTDataset.News2019,
                source_lang=Language.English,
                target_lang=Language.German,
                local_root="data/nlp/wmt",
                model_name="Facebook-FAIR (single)",
                paper_arxiv_id="1907.06616"
            )

            model = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model',
                force_reload=True, tokenizer='moses', bpe='fastbpe').cuda()

            for sid, text in tqdm(evaluator.source_segments.items()):
                translated = model.translate(text)
                evaluator.add({sid: translated})
                if evaluator.cache_exists:
                    break

            evaluator.save()
            print(evaluator.results)
    """

    task = "Machine Translation"

    _datasets = {
        (WMTDataset.News2014, Language.English, Language.German),
        (WMTDataset.News2019, Language.English, Language.German),
        (WMTDataset.News2014, Language.English, Language.French),
    }

    def __init__(self,
                 dataset: WMTDataset,
                 source_lang: Language,
                 target_lang: Language,
                 local_root: str = '.',
                 source_dataset_filename: str = None,
                 target_dataset_filename: str = None,
                 model_name: str = None,
                 paper_arxiv_id: str = None,
                 paper_pwc_id: str = None,
                 paper_results: dict = None,
                 model_description: str = None,
                 tokenization: Callable[[str], str] = None):
        """
        Creates an evaluator for one of the WMT benchmarks.

        :param dataset: Which dataset to evaluate on, f.e., WMTDataset.News2014.
        :param source_lang: Source language of the documents to translate.
        :param target_lang: Target language into which the documents are translated.
        :param local_root: Path to the directory where the dataset files are located locally.
            Ignored when run on sotabench server.
        :param source_dataset_filename: Local filename of the SGML file with the source documents.
            If None, the standard WMT filename is used, based on :param:`dataset`,
            :param:`source_lang` and :param:`target_lang`.
            Ignored when run on sotabench server.
        :param target_dataset_filename: Local filename of the SGML file with the reference documents.
            If None, the standard WMT filename is used, based on :param:`dataset`,
            :param:`source_lang` and :param:`target_lang`.
            Ignored when run on sotabench server.
        :param model_name: The name of the model from the
            paper - if you want to link your build to a model from a
            machine learning paper. See the WMT benchmarks pages for model names,
            (f.e., https://sotabench.com/benchmarks/machine-translation-on-wmt2014-english-german)
            on the paper leaderboard or models yet to try tabs.
        :param paper_arxiv_id: Optional linking to arXiv if you
            want to link to papers on the leaderboard; put in the
            corresponding paper's arXiv ID, e.g. '1907.06616'.
        :param paper_pwc_id: Optional linking to Papers With Code;
            put in the corresponding papers with code URL slug, e.g.
            'facebook-fairs-wmt19-news-translation-task'
        :param paper_results: If the paper model you are reproducing
            does not have model results on sotabench.com, you can specify
            the paper results yourself through this argument, where keys
            are metric names, values are metric values. e.g:

                    {'SacreBLEU': 42.7, 'BLEU score': 43.1}.

            Ensure that the metric names match those on the sotabench
            leaderboard - for WMT benchmarks it should be `SacreBLEU` for de-tokenized
            case sensitive BLEU score and `BLEU score` for tokenized BLEU.
        :param model_description: Optional model description.
        :param tokenization: An optional tokenization function to compute tokenized BLEU score.
            It takes a single string - a segment to tokenize, and returns a string with tokens
            separated by space, f.e.:

                    tokenization = lambda seg: seg.replace("'s", " 's").replace("-", " - ")

            If None, only de-tokenized SacreBLEU score is reported.
        """

        super().__init__(model_name, paper_arxiv_id, paper_pwc_id, paper_results, model_description)
        self.root = change_root_if_server(root=local_root,
                                          server_root=".data/nlp/wmt")
        self.dataset = dataset
        self.source_lang = source_lang
        self.target_lang = target_lang

        default_src_fn, default_dst_fn = self._get_source_dataset_filename()
        if source_dataset_filename is None or is_server():
            source_dataset_filename = default_src_fn

        if target_dataset_filename is None or is_server():
            target_dataset_filename = default_dst_fn

        self.source_dataset_path = Path(self.root) / source_dataset_filename
        self.target_dataset_path = Path(self.root) / target_dataset_filename

        self.metrics = TranslationMetrics(self.source_dataset_path, self.target_dataset_path, tokenization)

    def _get_source_dataset_filename(self):
        if self.dataset == WMTDataset.News2014:
            other_lang = self.source_lang.value if self.target_lang == Language.English else self.target_lang.value
            source = "{0}-{1}en-src.{2}.sgm".format(self.dataset.value, other_lang, self.source_lang.value)
            target = "{0}-{1}en-ref.{2}.sgm".format(self.dataset.value, other_lang, self.target_lang.value)
        elif self.dataset == WMTDataset.News2019:
            source = "{0}-{1}{2}-src.{1}.sgm".format(self.dataset.value, self.source_lang.value, self.target_lang.value)
            target = "{0}-{1}{2}-ref.{2}.sgm".format(self.dataset.value, self.source_lang.value, self.target_lang.value)
        else:
            raise ValueError("Unknown dataset: {}".format(self.dataset))
        return source, target

    def _get_dataset_name(self):
        cfg = (self.dataset, self.source_lang, self.target_lang)
        if cfg not in WMTEvaluator._datasets:
            raise ValueError("Unsupported dataset configuration: {} {} {}".format(
                self.dataset.name,
                self.source_lang.name,
                self.target_lang.name
            ))

        ds_names = {WMTDataset.News2014: "WMT2014", WMTDataset.News2019: "WMT2019"}
        return "{0} {1}-{2}".format(ds_names.get(self.dataset), self.source_lang.fullname, self.target_lang.fullname)

    def add(self, answers: Dict[str, str]):
        """
        Updates the evaluator with new results

        :param answers: a dict where keys are source segments ids and values are translated segments
            (segment id is created by concatenating document id and the original segment id,
            separated by `#`.)

        Examples:
            Update the evaluator with three results:

            .. code-block:: python

                my_evaluator.add({
                    'bbc.381790#1': 'Waliser AMs sorgen sich um "Aussehen wie Muppets"',
                    'bbc.381790#2': 'Unter einigen AMs herrscht Bestürzung über einen...',
                    'bbc.381790#3': 'Sie ist aufgrund von Plänen entstanden, den Namen...'
                })

        .. seealso:: `source_segments`
        """

        self.metrics.add(answers)

        if not self.first_batch_processed and self.metrics.has_data:
            self.batch_hash = calculate_batch_hash(
                self.cache_values(answers=self.metrics.answers,
                                  metrics=self.metrics.get_results(ignore_missing=True))
            )
            self.first_batch_processed = True

    @property
    def source_segments(self):
        """
        Ordered dictionary of all segments to translate with segments ids as keys. The same segments ids
        have to be used when submitting translations with :func:`add`.

        Examples:

            .. code-block:: python

                for segment_id, text in my_evaluator.source_segments.items():
                    translated = model(text)
                    my_evaluator.add({segment_id: translated})

        .. seealso: `source_documents`
        """

        return self.metrics.source_segments

    @property
    def source_documents(self):
        """
        List of all documents to translate

        Examples:

            .. code-block:: python

                for document in my_evaluator.source_documents:
                    for segment in document.segments:
                        translated = model(segment.text)
                        my_evaluator.add({segment.id: translated})

        .. seealso: `source_segments`
        """

        return self.metrics.source_documents

    def reset(self):
        """
        Removes already added translations

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

    def get_results(self):
        """
        Gets the results for the evaluator. Empty string is assumed for segments for which in translation
        was provided.

        :return: dict with `SacreBLEU` and `BLEU score`.
        """

        if self.cached_results:
            return self.results
        self.results = self.metrics.get_results()
        self.speed_mem_metrics['Max Memory Allocated (Total)'] = get_max_memory_allocated()

        return self.results

    def save(self):
        dataset = self._get_dataset_name()

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

