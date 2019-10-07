import time

from sotabenchapi.client import Client
from sotabenchapi.core import BenchmarkResult
from sotabencheval.utils import is_server
from sotabencheval.core.cache import cache_value


class BaseEvaluator:
    def __init__(self,
                 model_name: str = None,
                 paper_arxiv_id: str = None,
                 paper_pwc_id: str = None,
                 paper_results: dict = None,
                 model_description=None,):
        self.model_name = model_name
        self.paper_arxiv_id = paper_arxiv_id
        self.paper_pwc_id = paper_pwc_id
        self.paper_results = paper_results
        self.model_description = model_description

        self.first_batch_processed = False
        self.batch_hash = None
        self.cached_results = False
        self.results = None
        self._cache_exists = None

        self.init_time = time.time()
        self.speed_mem_metrics = {}

    @property
    def cache_exists(self):
        """
        Checks whether the cache exists in the sotabench.com database - if so
        then sets self.results to cached results and returns True.

        You can use this property for control flow to break a for loop over a dataset
        after the first iteration. This prevents rerunning the same calculation for the
        same model twice.

        Examples:
            Breaking a for loop

            .. code-block:: python

                ...

                with torch.no_grad():
                    for i, (input, target) in enumerate(iterator):
                        ...
                        output = model(input)
                        # optional formatting of output here to be a list of detection dicts
                        evaluator.add(output)

                        if evaluator.cache_exists:
                            break

                evaluator.save()

        :return: bool or None (if not in check mode)
        """

        if not is_server():  # we only check the cache on the server
            return None

        if not self.first_batch_processed:
            return False

        if self._cache_exists is not None:
            return self._cache_exists

        client = Client.public()
        cached_res = client.get_results_by_run_hash(self.batch_hash)
        if cached_res:
            self.results = cached_res
            self.cached_results = True
            print(
                "No model change detected (using the first batch run "
                "hash). Will use cached results."
            )
            self._cache_exists = True
        else:
            self._cache_exists = False
        return self._cache_exists

    def cache_values(self, **kwargs):
        return cache_value(kwargs)

    def reset_time(self):
        self.init_time = time.time()

    def save(self, **kwargs):
        """
        Calculate results and then put into a BenchmarkResult object

        On the sotabench.com server, this will produce a JSON file serialisation and results will be recorded
        on the platform.

        :return: BenchmarkResult object with results and metadata
        """

        # recalculate to ensure no mistakes made during batch-by-batch metric calculation
        self.get_results()

        return BenchmarkResult(
            task=self.task,
            config={},
            results=self.results,
            speed_mem_metrics=self.speed_mem_metrics,
            model=self.model_name,
            model_description=self.model_description,
            arxiv_id=self.paper_arxiv_id,
            pwc_id=self.paper_pwc_id,
            paper_results=self.paper_results,
            run_hash=self.batch_hash,
            **kwargs,
        )
