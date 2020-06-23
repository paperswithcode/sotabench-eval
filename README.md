<p align="center"><img width=500 src="/docs/docs/img/sotabencheval.png"></p>

--------------------------------------------------------------------------------

[![PyPI version](https://badge.fury.io/py/sotabencheval.svg)](https://badge.fury.io/py/sotabencheval) [![Generic badge](https://img.shields.io/badge/Documentation-Here-<COLOR>.svg)](https://paperswithcode.github.io/sotabench-eval/)

`sotabencheval` is a framework-agnostic library that contains a collection of deep learning benchmarks you can use to benchmark your models. It can be used in conjunction with the [sotabench](https://www.sotabench.com) service to record results for models, so the community can compare model performance on different tasks, as well as a continuous integration style service for your repository to benchmark your models on each commit.

## Benchmarks Supported

- [ADE20K](https://paperswithcode.github.io/sotabench-eval/ade20k/) (Semantic Segmentation)
- [COCO](https://paperswithcode.github.io/sotabench-eval/coco/) (Object Detection)
- [ImageNet](https://paperswithcode.github.io/sotabench-eval/imagenet/) (Image Classification)
- [SQuAD](https://paperswithcode.github.io/sotabench-eval/squad/) (Question Answering)
- [WikiText-103](https://paperswithcode.github.io/sotabench-eval/wikitext103/) (Language Modelling)
- [WMT](https://paperswithcode.github.io/sotabench-eval/wmt/) (Machine Translation)

PRs welcome for further benchmarks! 

## Installation

Requires Python 3.6+. 

```bash
pip install sotabencheval
```

## Get Benching! 🏋️

You should read the [full documentation here](https://paperswithcode.github.io/sotabench-eval/index.html), which contains guidance on getting started and connecting to [sotabench](https://www.sotabench.com).

Integration is lightweight. For example, if you are evaluating an ImageNet model, you initialize an Evaluator object and (optionally) link to any linked paper:

```python
from sotabencheval.image_classification import ImageNetEvaluator
evaluator = ImageNetEvaluator(
             model_name='FixResNeXt-101 32x48d',
             paper_arxiv_id='1906.06423')
```

Then for each batch of predictions your model makes on ImageNet, pass a dictionary of keys as image IDs and values as a `np.ndarray`s of logits to the `evaluator.add` method:

```python
evaluator.add(output_dict=dict(zip(image_ids, batch_output)))
```

The evaluation logic just needs to be written in a `sotabench.py` file and sotabench will run it on each commit and record the results:

<a href="https://sotabench.com/user/htvr/repos/TouvronHugo/FixRes#latest-results"><img width=500 src="/docs/docs/img/results.png"></a>

## Contributing

All contributions welcome!



