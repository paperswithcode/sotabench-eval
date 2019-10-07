# WikiText-103

![Banner](img/banner.png)
[//]: # (TODO Change Me) 

You can download WikiText-103 dataset [here](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/), there is no official leaderboard.

## TLDR;

Here is a minimal working example that let you evaluate `transformer-xl` from `huggingface/transformers`.

```python
import torch
from sotabencheval.langauge_modelling import WikiText103Evaluator

model = torch.hub.load('huggingface/transformers', 'modelWithLMHead', 'transfo-xl-wt103').to("cuda")
tokenizer = torch.hub.load('huggingface/transformers', 'tokenizer', 'transfo-xl-wt103').to("cuda")

with WikiText103Evaluator.dataset.get_path('local/dir/wikitext-103').open() as f:
    test_data = torch.tensor(tokenizer.encode(f.read())).to("cuda").eval()

def log_probs_generator(model, test_data, seq_len=128):
    with torch.no_grad():
        X, Y, mems = test_data[None, :-1], test_data[None, 1:], None
        for s in range(0, X.shape[-1], seq_len):
            log_probs, mems, *_ = model(input_ids=X[s:s+seq_len], mems=mems)
            yield log_probs, y[s:s+seq_len]

WikiText103Evaluator(
    model_name="Transformer-XL Large", 
    paper_arxiv_id="1901.02860",
    paper_pwc_id="transformer-xl-attentive-language-models" 
).eval(log_probs_generator(model, test_data)).print_results()
```
## Getting Started

You'll need the following in the root of your repository:

- `sotabench.py` file - contains benchmarking logic; the server will run this on each commit
- `requirements.txt` file - Python dependencies to be installed before running `sotabench.py`
- `sotabench_setup.sh` *(optional)* - any advanced dependencies or setup, e.g. compilation

You can write whatever you want in your `sotabench.py` file to get langauge model predictions on the WikiText-103 dataset.

But you will need to record your results for the server, and you'll want to avoid doing things like
downloading the dataset on the server. So you should:

- **Point to the server WikiText-103 data path** - popular datasets are pre-downloaded on the server.
- **Include an Evaluation object** in `sotabench.py` file to record the results.
- **Use Caching** *(optional)* - to speed up evaluation by hashing the first batch of predictions.

We explain how to do these various steps below.

## Server Data Location

The WikiText-103 development data is located in the root of your repository on the server at `.data/nlp/wikitext-103/wikitext-103-v1.zip`.
The zip file has default format of WikiText-103 and contains a folder `wikitext-103` with the following files:

- `wiki.train.tokens`
- `wiki.valid.tokens`
- `wiki.test.tokens`

Your local files may have a different file directory structure, so you
can use control flow like below to change the data path if the script is being
run on sotabench servers:

``` python
from sotabencheval.utils import is_server

if is_server():
    DATA_ROOT = '.data/nlp/squad'
else: # local settings
    DATA_ROOT = '/home/ubuntu/my_data/'
```
This will detect if `sotabench.py` is being run on the server and change behaviour accordingly.

Alternatively you can use our helper function that will automatically unpack the zip for you and point you to the right location as follows:

```python
from sotabencheval.langauge_modelling import WikiText103Evaluator

test_file_path = WikiText103Evaluator.dataset.get_path('/home/ubuntu/data')
```

## How Do I Initialize an Evaluator?

Add this to your code - before you start batching over the dataset and making predictions:

``` python
from sotabencheval.langauge_modelling import WikiText103Evaluator

evaluator = WikiText103Evaluator(model_name='Model name as found in paperswithcode website')
```

If you are reproducing a model from a paper, then you can enter the arXiv ID. If you
put in the same model name string as on the
[Wikitext-103](https://sotabench.com/benchmarks/language-modelling-on-wikitext-103) leaderboard
then you will enable direct comparison with the paper's model. If the `arxiv` is not avaliable you can use `paperswithcode.com` id.  Below is an example of an evaluator that matches `Transformer XL`:

``` python
from sotabencheval.langauge_modelling import WikiText103Evaluator

evaluator = WikiText103Evaluator(
    model_name="Transformer-XL Large",
    paper_arxiv_id="1901.02860",
    paper_pwc_id="transformer-xl-attentive-language-models"
)
```

The above will directly compare with the result of the paper when run on the server.

## How Do I Evaluate Predictions?

The evaluator object has an `.add(log_probabilites, labels)` method to submit predictions by batch or in full. The method except two `np.arrays` or `torch.tensors` of the same shape `[batch_size, sequence_length]`, first one holding `log` probabilities of each token and second holding token ids. 

If your model use subword tokenization you are free to use them without the need to convert them to full words, but make sure to mark that in your evaluator with `subword_tokenization=True`, as we are including some checks to make the perplexity calculation less error prone.

[//]: # (TODO Describe how to ensure that the right tokenization is being used) 

In order to save compute we use caching mechanism so it is recommended to output prediction in larger batches, (we cache results based on the first batch).

That would look something like this (for a PyTorch example):

``` python

evaluator = WikiText103Evaluator(
    model_name='GPT-2 Small',
    paper_pwc_id="language-models-are-unsupervised-multitask",
    text_transformation=True,
    subword_tokenization = True
)

# run you data preprocessing, in case of GPT-2 the preprocessing removes moses artifacts
with torch.no_grad():
    model.eval()
    for input, target in data_loader:
        output = model(input)
        log_probs = torch.LogSoftmax(output, dim=-1)
        target_log_probs = output.gether(-1, targets.unsqueeze(-1))
        evaluator.add(target_log_probs, target)
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
    for input, target in data_loader:
        # ...
        output = model(input)
        log_probs = #...
        evaluator.add(log_probs, target)

        if evaluator.cache_exists:
            break

evaluator.save()
```

If the hash is the same as in the server, we infer that the model hasn't changed, so
we simply return hashed results rather than running the whole evaluation again.

Caching is very useful if you have large models, or a repository that is evaluating
multiple models, as it speeds up evaluation significantly.

Evaluator have a convenience method `eval` that encapsulates the above logic,  all you have to do is provide it with a generator that is able to output one batch at a time. The above code would look as follows:
```python
    def get_predictions(data_loader):
        with torch.no_grad():
        for input, target in data_loader:
            # ...
            output = model(input)
            log_probs = #...
            yield log_probs, target # instead of evaluator.add(...)

    evaluator.eval(get_predictions(data_loader))
```

## Need More Help?

Head on over to the [Natural Language Processing](https://forum.sotabench.com/c/nlp) section of the sotabench forums if you have any questions or difficulties.
