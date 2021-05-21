from sotabencheval.core import BaseEvaluator
from sotabencheval.utils import calculate_batch_hash, change_root_if_server, is_server
from sotabencheval.utils import get_max_memory_allocated
from typing import Dict, Callable
from pathlib import Path
from enum import Enum
from sacrebleu import corpus_bleu
import time
import os

MIN_CACHE_BATCH_SIZE = 32

class TatoebaDataset(Enum):
    v1 = "v2020-07-28"
    
class TatoebaEvaluator(BaseEvaluator):
    """Evaluator for Tatoeba benchmarks.

    Examples:
        Evaluate a Transformer model from the fairseq repository on a given language pair:

        .. code-block:: python

            from sotabencheval.machine_translation import TatoebaEvaluator
            from tqdm import tqdm
            import torch

            evaluator = TatoebaEvaluator(
                dataset=TatoebaDataset.v1
                source_lang="eng",
                target_lang="deu",
                local_root="data/tatoeba/test",
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

    def __init__(self,
                 dataset: TatoebaDataset,
                 source_lang: str,
                 target_lang: str,
                 local_root: str = '.',
                 model_name: str = None,
                 paper_arxiv_id: str = None,
                 paper_pwc_id: str = None,
                 paper_results: dict = None,
                 model_description: str = None,
                 tokenization: Callable[[str], str] = None):
        """
        Creates an evaluator for one of the Tatoeba test corpora.

        :param dataset: Which dataset to evaluate on, f.e., TatoebaDataset.v1.
        :param source_lang: Source language of the documents to translate.
        :param target_lang: Target language into which the documents are translated.
        :param local_root: Path to the directory where the dataset files are located locally.
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
                                          server_root=".data/nlp/tatoeba")

        self.dataset = dataset
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.dataset_path = self._get_dataset_filename(self.root)
        self._source_col, self._target_col = self._get_data_columns()
        self._load_dataset() # This is where the reference data gets loaded
        self.answers = [None] * len(self.targets) # The model-submitted translations
        self._tokenization = tokenization

    def _get_dataset_filename(self, root):
        if not os.path.isdir(root):
            raise FileNotFoundError("Couldn't access root directory for Tatoeba test data")
        if self.dataset == TatoebaDataset.v1:
            path = os.path.join("test-v2020-07-28", "-".join(sorted((self.source_lang, self.target_lang))), "test.txt")
            if not os.path.isfile(os.path.join(root, path)):
                raise FileNotFoundError(f"Pair {self.source_lang}-{self.target_lang} not found in given Tatoeba test data directory")
            return Path(self.root) / path
        else:
            raise ValueError("Unknown dataset: {}".format(self.dataset))
        
    def _get_data_columns(self):
        if self.dataset == TatoebaDataset.v1:
            if self.source_lang < self.target_lang:
                return (2, 3)
            else:
                return (3, 2)
        else:
            raise ValueError("Unknown dataset: {}".format(self.dataset))


    def add(self, answers: Dict[int, str]):
        """
        Updates the evaluator with new results

        :param answers: a dict where keys are sentence ids and values are translated segments

        Examples:
            Update the evaluator with three results:

            .. code-block:: python

                my_evaluator.add({
                    4: 'Waliser AMs sorgen sich um "Aussehen wie Muppets"',
                    89: 'Unter einigen AMs herrscht Bestürzung über einen...',
                    123: 'Sie ist aufgrund von Plänen entstanden, den Namen...'
                })

        """

        if len(answers) == 0:
            print("Empty batch added to results")
            return
        for k, v in answers.items():
            # zero-based indexing
            k = k - 1
            if k < 0 or k >= len(self.answers):
                print(f"Tried to add result with invalid key {k+1}, must be between 1 and {len(self.answers)}")
                continue
            if self.answers[k] != None:
                print("Multiple translations for the same segment")
            self.answers[k] = v
        if not self.first_batch_processed and self.has_data:
            self.batch_hash = calculate_batch_hash(
                self.cache_values(answers=self.answers,
                                  metrics=self.get_results(ignore_missing=True))
            )
            self.first_batch_processed = True

    def _load_dataset(self):
        self.sources = []
        self.targets = []
        for line in open(self.dataset_path):
            parts = line.strip().split('\t')
            self.sources.append(parts[self._source_col])
            self.targets.append(parts[self._target_col])
            
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

        self.answers = [None] * len(self.answers)

    def evaluate(self, ignore_missing=False):
        if ignore_missing:
            answers = [self.answers[i] for i in range(len(self.answers)) if self.answers[i] != None]
            references = [self.targets[i] for i in range(len(self.answers)) if self.answers[i] != None]
        else:
            answers = [self.answers[i] if self.answers[i] != None else "" for i in range(len(self.answers))]
            references = self.targets[:]
        bleu = corpus_bleu(answers, [references])
        self.results = {'SacreBLEU': bleu.score}
        if self._tokenization is not None:
            tok_answers = [self._tokenization(answer) for answer in answers]
            tok_references = [self._tokenization(reference) for reference in references]
            tok_bleu = corpus_bleu(tok_answers, [tok_references], tokenize='none', force=True)
            self.results['BLEU score'] = tok_bleu.score

    @property
    def has_data(self):
        return len([a for a in self.answers if a != None]) >= MIN_CACHE_BATCH_SIZE

    def get_results(self, ignore_missing=False):
        """
        Gets the results for the evaluator.

        :return: dict with `SacreBLEU` and `BLEU score`.
        """

        if self.cached_results:
            return self.results
        self.evaluate(ignore_missing)
        try:
            self.speed_mem_metrics['Max Memory Allocated (Total)'] = get_max_memory_allocated()
        except:
            self.speed_mem_metrics['Max Memory Allocated (Total)'] = None

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

