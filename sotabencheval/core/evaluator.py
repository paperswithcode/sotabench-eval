import time

from sotabenchapi.client import Client
from sotabenchapi.core import BenchmarkResult
from sotabencheval.utils import is_server
from sotabencheval.core.cache import cache_value


class BaseEvaluator:
    """Base class for evaluator objects on tasks

    Currently SQuAD and WMT use this as a parent.

    TODO: Refactor ImageNet, COCO, ADE20K, PASCAL to utilise this class

    The core API design relies upon:

    (a) Initializing an Evaluator object and linking to a paper, for example:

    .. code-block:: python

        from sotabencheval.question_answering import SQuADEvaluator, SQuADVersion

        evaluator = SQuADEvaluator(model_name='SpanBERT', paper_arxiv_id='1907.10529',
            version=SQuADVersion.V20)

    The paper metadata allows the results to be linked to paper results when submitted to sotabench.com.

    (b) Adding Predictions (usually in batch) - example below for PyTorch iterating over DataLoader:

    .. code-block:: python

            for i, (input, target) in enumerate(data_loader):
                ...
                output = model(input)
                # potentially formatting of the output here
                evaluator.add(output)

    These results are accumulated and then evaluated - i.e. metrics are calculated once done.

    (c) Saving Results

    .. code-block:: python
        evaluator.save()

    Gets the evaluation results for the current predictions added to the Evaluation object - calculates metrics -
    then run if on the server, serializes results to a sotabench_results.json file which is processed and results
    are stored on the server.

    These three steps: initialization -> adding predictions -> saving and evaluating results are the core API.
    They should be capable of integration with any existing evaluation logic in your repository.
    """

    def __init__(self,
                 model_name: str = None,
                 paper_arxiv_id: str = None,
                 paper_pwc_id: str = None,
                 paper_results: dict = None,
                 model_description=None,):
        """
        Initializes a BaseEvaluator like object

        :param model_name: (str) The name of the model, for example 'ResNet-101', which will be saved to sotabench.com
        :param paper_arxiv_id: (str, optional) The paper that the model is linked to, e.g. '1906.06423'
        :param paper_pwc_id: (str, optional) The PWC paper id (slug), e.g. 'albert-a-lite-bert-for-self-supervised'
        :param paper_results: (dict, optional) If the paper you are linking to does not have results on sotabench,
        then you can add paper results here. This will be a dictionary with keys as metric names, and values as metric
        values. This will be benchmark specific.
        :param model_description: (str, optional) Optional description for the model; this can contain details about
        where the weights are from, details about training, and more. This will appear in an info box for the model
        when it is displayed on sotabench.com.
        """

        # Model metadata

        self.model_name = model_name
        self.paper_arxiv_id = paper_arxiv_id
        self.paper_pwc_id = paper_pwc_id
        self.paper_results = paper_results
        self.model_description = model_description

        # Backend variables for hashing and caching

        self.first_batch_processed = False
        self.batch_hash = None
        self.cached_results = False
        self.results = None
        self._cache_exists = None

        # Speed and memory metrics

        self.init_time = time.time()
        self.speed_mem_metrics = {}

    @property
    def cache_exists(self):
        """
        Checks whether the cache exists in the sotabench.com database - if so
        then sets self.results to cached results and returns True.

        You can use this property for control flow to break a for loop over a dataset
        after the first iteration. This prevents re-running the same calculation for the
        same model twice.

        Q: Why should the user use this?
        A: If you want fast "continuous evaluation" and don't want to avoid rerunning the same model over and over
            each time you commit something new to your repository.

        Examples:
            Breaking a for loop if the model is the same as last time we ran

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

        This logic is for the server; it will not break the loop if you evaluate locally.

        :return: bool or None (if not on server)
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
                f"hash {self.batch_hash}). Will use cached results."
            )

            self._cache_exists = True
        else:
            self._cache_exists = False
        return self._cache_exists

    def reset(self):
        """Resets the internal state of evaluator and allows to start over"""
        pass

    def cache_values(self, **kwargs):
        """
        Takes in keyword argument and converts to a hashable (cachable) format for each

        :param kwargs: keyword argument
        :return: cachable version of the keyword arguments
        """
        return cache_value(kwargs)

    def eval(self, results_generator):
        """Run full evaluation loop on results_genertor"""
        self.reset()
        self.reset_time()
        for results in results_generator:
            self.add(*results)
            if self.first_batch_processed and self.cache_exists:
                break
        self.save()
        return self
    
    def get_results(self):
        """Calculate results."""
        return self.results

    def print_results(self):
        """Print results."""
        self.get_results()
        print(f"results = {self.results}, speed_mem_metrics = {self.speed_mem_metrics}")

    def reset_time(self):
        """
        Simple method to reset the timer self.init_time. Often used before a loop, to time the evaluation
        appropriately, for example:

        .. code-block:: python

            from sotabencheval.question_answering import SQuADEvaluator, SQuADVersion

            evaluator = SQuADEvaluator(model_name='SpanBERT', paper_arxiv_id='1907.10529',
                version=SQuADVersion.V20)

            # processing/setup logic here

            evaluator.reset_time()

            for i, (input, target) in enumerate(data_loader):
                ...
                output = model(input)
                # potentially formatting of the output here
                evaluator.add(output)

            evaluator.save()

        Above we may have processing logic inbetween the evaluator initialization and the actual evaluation loop, so
        we reset the timer so it's a fair timing of the evaluation (and not setup steps like data processing, loading
        the model etc).

        :return: void - resets self.init_time
        """
        self.init_time = time.time()

    def save(self, **kwargs):
        """
        Calculate results and then put into a BenchmarkResult object

        On the sotabench.com server, this will produce a JSON file serialisation in sotabench_results.json and results
        will be recorded on the platform.

        Users should save once all predictions are added, for instance:

        .. code-block:: python

            from sotabencheval.question_answering import SQuADEvaluator, SQuADVersion

            evaluator = SQuADEvaluator(model_name='SpanBERT', paper_arxiv_id='1907.10529',
                version=SQuADVersion.V20)

            # processing/setup logic here

            evaluator.reset_time()

            for i, (input, target) in enumerate(data_loader):
                ...
                output = model(input)
                # potentially formatting of the output here
                evaluator.add(output)

            evaluator.save()

        Here once we have added all the predictions to the evaluator, we .save() so we evaluate and, if on the server,
        results are serialized and saved to the server.

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
