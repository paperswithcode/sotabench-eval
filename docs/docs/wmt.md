# WMT

You can view the WMT Machine Translation leaderboards:

- [WMT2014 English-German](https://sotabench.com/benchmarks/machine-translation-on-wmt2014-english-german)
- [WMT2014 English-French](https://sotabench.com/benchmarks/machine-translation-on-wmt2014-english-french)
- [WMT2019 English-German](https://sotabench.com/benchmarks/machine-translation-on-wmt2019-english-german)

## Getting Started

You'll need the following in the root of your repository:

- `sotabench.py` file - contains benchmarking logic; the server will run this on each commit
- `requirements.txt` file - Python dependencies to be installed before running `sotabench.py`
- `sotabench_setup.sh` *(optional)* - any advanced dependencies or setup, e.g. compilation

You can write whatever you want in your `sotabench.py` file to get model predictions on the WMT datasets.

But you will need to record your results for the server, and you'll want to avoid doing things like
downloading the dataset on the server. So you should:

- **Include an Evaluation object** in `sotabench.py` file to record the results.
- **Point to the server WMT data path** - popular datasets are pre-downloaded on the server.
- **Use Caching** *(optional)* - to speed up evaluation by hashing the first batch of predictions.

We explain how to do these various steps below.

## How Do I Initialize an Evaluator?

Before you start batching over the dataset and making predictions you need
to create an evaluator instance to record results for a given leaderboard.
For example, to evaluate on WMT2014 News English-French test set add this
to your code:

``` python
from sotabencheval.machine_translation import WMTEvaluator, WMTDataset, Language

evaluator = WMTEvaluator(
    dataset=WMTDataset.News2014,
    source_lang=Language.English,
    target_lang=Language.French,
    local_root='mydatasets',
    model_name='My Super Model'
)
```

You can use `evaluator.source_dataset_path: Path` and `evaluator.target_dataset_path: Path`
to get paths to the source and target SGML files.
In the example above the first one resolves to `.data/nlp/wmt/newstest2014-fren-src.en.sgm` on
sotabench server and `mydatasets/newstest2014-fren-src.en.sgm` when run locally.
If you want to use non-standard file names locally you can override the defaults like this:

``` python
evaluator = WMTEvaluator(
    ...,
    local_root='mydatasets'
    source_dataset_filename='english.sgm',
    target_dataset_filename='french.sgm'
)
```

If you are reproducing a model from a paper, then you can enter the arXiv ID. If you
put in the same model name string as on the leaderboard
then you will enable direct comparison with the paper's model. For example:

``` python
evaluator = WMTEvaluator(
    dataset=WMTDataset.News2019,
    source_lang=Language.English,
    target_lang=Language.German,
    local_root="mydatasets",
    model_name="Facebook-FAIR (single)",
    paper_arxiv_id="1907.06616"
)
```

The above will directly compare with the result of the paper when run on the server.

Instead of parsing the dataset files by yourself you can access raw segments as strings:

``` python
    for segment_id, text in evaluator.source_segments:
        # translate text

    # or get segments within document context
    for document in evaluator.source_documents:
        context = [segment.text for segment in document.segments]
        for segment in document.segments:
            segment_id, text = segment.id, segment.text
            # translate text in context
```

## How Do I Evaluate Predictions?

The evaluator object has an `.add(answers: Dict[str, str])` method to submit predictions by batch or in full.

For WMT the expected input is a dictionary, where keys are source segments
ids and values are translated segments
(segment id is created by concatenating document id and the original segment id,
separted by `#`.) For example:

``` python
evaluator.add({
    'bbc.381790#1': 'Waliser AMs sorgen sich um "Aussehen wie Muppets"',
    'bbc.381790#2': 'Unter einigen AMs herrscht Bestürzung über einen...',
    'bbc.381790#3': 'Sie ist aufgrund von Plänen entstanden, den Namen...'
})
```

You can do this all at once in a single call to `add()`, but more naturally, you will
probably loop over the dataset and call the method for the outputs of each batch.
That would look something like this (for a PyTorch example):

``` python
with torch.no_grad():
    for i, (input, target) in enumerate(data_loader):
        ...
        output = model(input)
        # potentially formatting of the output here to be a dict
        evaluator.add(output)
```

When you are done, you can get the results locally by running:

``` python
evaluator.get_results()
```

But for the server you want to save the results by running:

``` python
evaluator.save()
```

This method serialises the results and model metadata and stores to the server database.

## How Do I Cache Evaluation?

Sotabench reruns your script on every commit. This is good because it acts like
continuous integration in checking for bugs and changes, but can be annoying
if the model hasn't changed and evaluation is lengthy.

Fortunately sotabencheval has caching logic that you can use.

The idea is that after the first batch, we hash the model outputs and the
current metrics and this tells us if the model is the same given the dataset.
You can include hashing within an evaluation loop like follows (in the following
example for a PyTorch repository):

``` python
with torch.no_grad():
    for i, (input, target) in enumerate(data_loader):
        ...
        output = model(input)
        # potentially formatting of the output here to be a list of dicts
        evaluator.add(output)

        if evaluator.cache_exists:
            break

evaluator.save()
```

If the hash is the same as in the server, we infer that the model hasn't changed, so
we simply return hashed results rather than running the whole evaluation again.

Caching is very useful if you have large models, or a repository that is evaluating
multiple models, as it speeds up evaluation significantly.

## A Full sotabench.py Example

Below we show an implementation for a model from the torchhub repository. This
incorporates all the features explained above: (a) using the WMT Evaluator,
(b) accessing segments from evaluator, and (c) the evaluation caching logic.

``` python
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
    tokenizer='moses', bpe='fastbpe').cuda()

for sid, text in tqdm(evaluator.metrics.source_segments.items()):
    translated = model.translate(text)
    evaluator.add({sid: translated})
    if evaluator.cache_exists:
        break

evaluator.save()
print(evaluator.results)

```

## Need More Help?

Head on over to the [Natural Language Processing](https://forum.sotabench.com/c/nlp) section of the sotabench
forums if you have any questions or difficulties.
