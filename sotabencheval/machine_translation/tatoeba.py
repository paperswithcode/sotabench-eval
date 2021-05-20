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

    _v1_pairs = (
        ('lat', 'por'),
        ('eus', 'jpn'),
        ('eng', 'epo'),
        ('nor', 'rus'),
        ('deu', 'isl'),
        ('eng', 'est'),
        ('hun', 'swe'),
        ('eng', 'ota'),
        ('lit', 'rus'),
        ('eng', 'mar'),
        ('eng', 'jbo'),
        ('ara', 'ell'),
        ('deu', 'eng'),
        ('ben', 'eng'),
        ('hun', 'jpn'),
        ('bel', 'eng'),
        ('ina', 'tur'),
        ('heb', 'tur'),
        ('epo', 'ron'),
        ('deu', 'tat'),
        ('por', 'tur'),
        ('bul', 'spa'),
        ('lit', 'tur'),
        ('bel', 'ita'),
        ('eng', 'ile'),
        ('fra', 'ina'),
        ('nld', 'toki'),
        ('afr', 'deu'),
        ('kaz', 'rus'),
        ('eng', 'toki'),
        ('hun', 'por'),
        ('ell', 'fra'),
        ('ces', 'fra'),
        ('deu', 'rus'),
        ('epo', 'ita'),
        ('fra', 'zho'),
        ('deu', 'heb'),
        ('slv', 'zho'),
        ('eng', 'uzb'),
        ('ita', 'rus'),
        ('dan', 'swe'),
        ('ara', 'ber'),
        ('eus', 'spa'),
        ('ell', 'rus'),
        ('eng', 'pmn'),
        ('ita', 'lat'),
        ('ara', 'deu'),
        ('jpn', 'nor'),
        ('ina', 'nld'),
        ('fra', 'swe'),
        ('heb', 'jpn'),
        ('ber', 'eng'),
        ('bre', 'fra'),
        ('ces', 'epo'),
        ('ita', 'por'),
        ('ces', 'hun'),
        ('nld', 'rus'),
        ('fin', 'swe'),
        ('cor', 'ita'),
        ('tur', 'ukr'),
        ('ber', 'epo'),
        ('epo', 'lfn'),
        ('fry', 'nld'),
        ('hun', 'rus'),
        ('jbo', 'spa'),
        ('ces', 'deu'),
        ('eng', 'fao'),
        ('eng', 'lit'),
        ('ell', 'eng'),
        ('swe', 'zho'),
        ('ita', 'zho'),
        ('eng', 'lad'),
        ('eng', 'hun'),
        ('afr', 'nld'),
        ('eng', 'run'),
        ('eng', 'nds'),
        ('deu', 'pol'),
        ('deu', 'fin'),
        ('eng', 'ita'),
        ('ces', 'slv'),
        ('eng', 'hin'),
        ('epo', 'rus'),
        ('deu', 'slv'),
        ('ara', 'epo'),
        ('ron', 'spa'),
        ('heb', 'lat'),
        ('por', 'rus'),
        ('deu', 'ron'),
        ('eng', 'kha'),
        ('eng', 'spa'),
        ('fin', 'ita'),
        ('eng', 'tat'),
        ('fin', 'hun'),
        ('cha', 'eng'),
        ('fra', 'hun'),
        ('jpn', 'rus'),
        ('spa', 'swe'),
        ('ina', 'rus'),
        ('deu', 'fas'),
        ('ido', 'spa'),
        ('spa', 'vie'),
        ('epo', 'vie'),
        ('fin', 'kor'),
        ('eng', 'mri'),
        ('dan', 'tur'),
        ('eng', 'khm'),
        ('jpn', 'msa'),
        ('pol', 'rus'),
        ('jpn', 'por'),
        ('por', 'tgl'),
        ('eng', 'pol'),
        ('isl', 'spa'),
        ('ces', 'rus'),
        ('eng', 'ltz'),
        ('ell', 'epo'),
        ('run', 'rus'),
        ('eng', 'ido'),
        ('bul', 'fra'),
        ('hun', 'pol'),
        ('deu', 'hun'),
        ('dan', 'epo'),
        ('epo', 'hbs'),
        ('cat', 'ukr'),
        ('jpn', 'nld'),
        ('epo', 'nld'),
        ('epo', 'spa'),
        ('deu', 'ltz'),
        ('ita', 'ron'),
        ('fra', 'ron'),
        ('por', 'toki'),
        ('pol', 'spa'),
        ('deu', 'nor'),
        ('jpn', 'lit'),
        ('epo', 'nor'),
        ('gos', 'nld'),
        ('eng', 'fra'),
        ('eng', 'jav'),
        ('fin', 'fkv'),
        ('cat', 'nld'),
        ('eng', 'jpn'),
        ('tlh', 'zho'),
        ('ces', 'eng'),
        ('orv', 'ukr'),
        ('cat', 'eng'),
        ('epo', 'lit'),
        ('cbk', 'eng'),
        ('awa', 'eng'),
        ('epo', 'tgl'),
        ('deu', 'jbo'),
        ('fra', 'heb'),
        ('heb', 'ukr'),
        ('eng', 'gos'),
        ('epo', 'ido'),
        ('epo', 'heb'),
        ('eng', 'tlh'),
        ('ina', 'spa'),
        ('nds', 'nld'),
        ('chv', 'rus'),
        ('eng', 'zza'),
        ('ido', 'ita'),
        ('epo', 'fin'),
        ('vie', 'zho'),
        ('ota', 'tur'),
        ('ell', 'swe'),
        ('ron', 'tur'),
        ('fra', 'rus'),
        ('deu', 'toki'),
        ('ara', 'spa'),
        ('hun', 'ita'),
        ('deu', 'kab'),
        ('eng', 'ilo'),
        ('heb', 'rus'),
        ('deu', 'epo'),
        ('ita', 'swe'),
        ('dan', 'spa'),
        ('deu', 'hbs'),
        ('lat', 'spa'),
        ('cat', 'spa'),
        ('rus', 'tat'),
        ('por', 'swe'),
        ('cor', 'fra'),
        ('hun', 'spa'),
        ('epo', 'tur'),
        ('eng', 'nor'),
        ('dan', 'nld'),
        ('fra', 'spa'),
        ('deu', 'swg'),
        ('fra', 'ita'),
        ('afr', 'eng'),
        ('deu', 'run'),
        ('eng', 'tgl'),
        ('por', 'zho'),
        ('deu', 'ido'),
        ('ber', 'fra'),
        ('uig', 'zho'),
        ('fra', 'vie'),
        ('cor', 'epo'),
        ('fin', 'nor'),
        ('nor', 'swe'),
        ('eng', 'urd'),
        ('lat', 'nor'),
        ('heb', 'pol'),
        ('eng', 'pms'),
        ('eng', 'swe'),
        ('eng', 'orv'),
        ('jpn', 'ukr'),
        ('hun', 'tur'),
        ('fra', 'slv'),
        ('jpn', 'spa'),
        ('eng', 'ron'),
        ('fra', 'tur'),
        ('est', 'rus'),
        ('avk', 'spa'),
        ('fra', 'uig'),
        ('chm', 'rus'),
        ('lat', 'yid'),
        ('eng', 'lfn'),
        ('heb', 'ita'),
        ('ces', 'ukr'),
        ('fin', 'fra'),
        ('fra', 'hbs'),
        ('run', 'spa'),
        ('lav', 'rus'),
        ('eng', 'tuk'),
        ('chv', 'eng'),
        ('nld', 'tur'),
        ('fra', 'jpn'),
        ('deu', 'ell'),
        ('eng', 'msa'),
        ('jpn', 'swe'),
        ('lfn', 'por'),
        ('lat', 'rus'),
        ('deu', 'hsb'),
        ('deu', 'lit'),
        ('spa', 'zho'),
        ('bul', 'tur'),
        ('fra', 'pcd'),
        ('hbs', 'ita'),
        ('jpn', 'kor'),
        ('fra', 'tgl'),
        ('ita', 'pol'),
        ('fin', 'jpn'),
        ('fra', 'jbo'),
        ('rus', 'spa'),
        ('eng', 'uig'),
        ('hun', 'ukr'),
        ('kor', 'spa'),
        ('cat', 'ita'),
        ('fra', 'kab'),
        ('bel', 'epo'),
        ('epo', 'lat'),
        ('eng', 'fin'),
        ('cat', 'fra'),
        ('fra', 'msa'),
        ('hbs', 'rus'),
        ('pol', 'zho'),
        ('ara', 'heb'),
        ('epo', 'hun'),
        ('eng', 'que'),
        ('deu', 'por'),
        ('deu', 'ukr'),
        ('fra', 'gcf'),
        ('bre', 'eng'),
        ('epo', 'isl'),
        ('msa', 'zho'),
        ('epo', 'fas'),
        ('ell', 'spa'),
        ('deu', 'yid'),
        ('eng', 'sqi'),
        ('ukr', 'zho'),
        ('ina', 'por'),
        ('nld', 'ukr'),
        ('isl', 'ita'),
        ('eng', 'isl'),
        ('bul', 'eng'),
        ('khm', 'spa'),
        ('deu', 'spa'),
        ('jpn', 'zho'),
        ('deu', 'swe'),
        ('heb', 'spa'),
        ('rus', 'toki'),
        ('hye', 'rus'),
        ('rus', 'sah'),
        ('bul', 'ita'),
        ('afr', 'spa'),
        ('hbs', 'nor'),
        ('isl', 'jpn'),
        ('ara', 'ita'),
        ('ces', 'ita'),
        ('fin', 'lat'),
        ('deu', 'tur'),
        ('deu', 'eus'),
        ('rus', 'swe'),
        ('dtp', 'eng'),
        ('deu', 'lat'),
        ('dan', 'eng'),
        ('eng', 'tur'),
        ('pol', 'swe'),
        ('bel', 'ukr'),
        ('bub', 'rus'),
        ('eng', 'vol'),
        ('bel', 'zho'),
        ('nld', 'ron'),
        ('nld', 'por'),
        ('fra', 'por'),
        ('hbs', 'zho'),
        ('por', 'ukr'),
        ('ita', 'spa'),
        ('fin', 'kur'),
        ('dan', 'rus'),
        ('epo', 'zho'),
        ('bul', 'ukr'),
        ('fra', 'pol'),
        ('rus', 'zho'),
        ('aze', 'eng'),
        ('ita', 'msa'),
        ('nor', 'nor'),
        ('ber', 'spa'),
        ('rus', 'slv'),
        ('ina', 'lat'),
        ('ltz', 'nld'),
        ('deu', 'tgl'),
        ('bel', 'pol'),
        ('eng', 'kat'),
        ('deu', 'jpn'),
        ('deu', 'tlh'),
        ('ara', 'rus'),
        ('deu', 'nds'),
        ('eng', 'gle'),
        ('eng', 'por'),
        ('hin', 'urd'),
        ('tat', 'vie'),
        ('epo', 'nds'),
        ('epo', 'fra'),
        ('ell', 'tur'),
        ('cat', 'por'),
        ('eng', 'swa'),
        ('aze', 'tur'),
        ('eng', 'heb'),
        ('eng', 'mkd'),
        ('ceb', 'eng'),
        ('ita', 'nld'),
        ('glg', 'spa'),
        ('afr', 'epo'),
        ('deu', 'ita'),
        ('fin', 'spa'),
        ('fin', 'rus'),
        ('rus', 'uig'),
        ('epo', 'toki'),
        ('ita', 'jpn'),
        ('eng', 'kor'),
        ('ita', 'tur'),
        ('epo', 'yid'),
        ('hin', 'zho'),
        ('jpn', 'vie'),
        ('lat', 'nld'),
        ('dan', 'fra'),
        ('eng', 'vie'),
        ('hun', 'zho'),
        ('tur', 'uig'),
        ('fra', 'nld'),
        ('epo', 'jbo'),
        ('nld', 'spa'),
        ('deu', 'kur'),
        ('spa', 'ukr'),
        ('heb', 'nld'),
        ('spa', 'tur'),
        ('rus', 'tur'),
        ('lit', 'pol'),
        ('msa', 'spa'),
        ('epo', 'swe'),
        ('hun', 'nld'),
        ('ces', 'spa'),
        ('deu', 'nld'),
        ('deu', 'zho'),
        ('spa', 'toki'),
        ('eng', 'fas'),
        ('pol', 'ukr'),
        ('ell', 'por'),
        ('eng', 'kur'),
        ('eng', 'hye'),
        ('dan', 'nor'),
        ('rus', 'tlh'),
        ('por', 'spa'),
        ('jpn', 'pol'),
        ('eng', 'ina'),
        ('dan', 'jpn'),
        ('ita', 'ukr'),
        ('deu', 'fra'),
        ('spa', 'tgl'),
        ('epo', 'glg'),
        ('nld', 'zho'),
        ('eng', 'hbs'),
        ('rus', 'ukr'),
        ('fas', 'fra'),
        ('jpn', 'tlh'),
        ('eng', 'nld'),
        ('ita', 'pms'),
        ('eng', 'lav'),
        ('ara', 'eng'),
        ('deu', 'ile'),
        ('fra', 'ukr'),
        ('afr', 'rus'),
        ('jbo', 'swe'),
        ('ara', 'jpn'),
        ('ina', 'lfn'),
        ('cor', 'eng'),
        ('lit', 'spa'),
        ('eng', 'eus'),
        ('eus', 'rus'),
        ('avk', 'fra'),
        ('eng', 'tha'),
        ('eng', 'war'),
        ('fra', 'nor'),
        ('epo', 'ina'),
        ('kor', 'zho'),
        ('epo', 'oci'),
        ('fin', 'tur'),
        ('nld', 'pol'),
        ('eng', 'grc'),
        ('ron', 'rus'),
        ('eng', 'lat'),
        ('jpn', 'nds'),
        ('epo', 'vol'),
        ('dan', 'deu'),
        ('epo', 'por'),
        ('lat', 'ukr'),
        ('gla', 'spa'),
        ('deu', 'msa'),
        ('fra', 'run'),
        ('deu', 'ina'),
        ('ina', 'yid'),
        ('eng', 'slv'),
        ('ara', 'tur'),
        ('fra', 'lat'),
        ('eng', 'zho'),
        ('epo', 'pol'),
        ('ita', 'lit'),
        ('eng', 'yid'),
        ('eng', 'ukr'),
        ('kat', 'rus'),
        ('epo', 'jpn'),
        ('epo', 'ukr'),
        ('ell', 'ita'),
        ('epo', 'tlh'),
        ('eng', 'mon'),
        ('hun', 'lat'),
        ('slv', 'ukr'),
        ('ina', 'ita'),
        ('deu', 'kor'),
        ('fin', 'por'),
        ('eng', 'kaz'),
        ('ara', 'fra'),
        ('eng', 'glg'),
        ('bel', 'fra'),
        ('tur', 'zho'),
        ('jbo', 'rus'),
        ('dan', 'fin'),
        ('eng', 'kab'),
        ('bel', 'rus'),
        ('eng', 'rus'),
        ('zho', 'zho'),
        ('nds', 'rus'),
        ('bul', 'rus'),
        )

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
        self.dataset_path = Path(self.root) / self._get_dataset_filename()
        self._source_col, self._target_col = self._get_data_columns()
        self._load_dataset() # This is where the reference data gets loaded
        self.answers = [None] * len(self.targets) # The model-submitted translations
        self._tokenization = tokenization

    def _get_dataset_filename(self):
        if self.dataset == TatoebaDataset.v1:
            return os.path.join("test-v2020-07-28", "-".join(sorted((self.source_lang, self.target_lang))), "test.txt")
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
            k = k - 1
            if k < 0 or k >= len(answers):
                print(f"Tried to add result with invalid key {k+1}, must be between 0 and {len(answers)}")
                continue
            if self.answers[k-1] != None:
                print("Multiple translations for the same segment")
            self.answers[k-1] = v
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

