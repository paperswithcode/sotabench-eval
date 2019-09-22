# Welcome to sotabencheval!

![SotaBench](img/banner.png)

You have reached the docs for the **sotabencheval** library. This library contains a collection of deep learning benchmarks you can use to
benchmark your models. It can be used in conjunction with the 
[sotabench](http://www.sotabench.com) website to record results for models, so the community
can compare model performance on different tasks, as well as a continuous integration style
service for your repository to benchmark your models on each commit.

**sotabencheval** is a general benchmarking library, meaning it is designed to support all deep learning frameworks, 
and requires minimal code integration. There are alternative sotabench APIs you can use that are
specialized for particular frameworks, e.g. [torchbench](https://github.com/paperswithcode/torchbench) for PyTorch.


## Getting Started : Benchmarking on ImageNet

**Step One : Create a sotabench.py file in the root of your repository**

This can contain whatever logic you need to load and process the dataset, and to 
produce model predictions for it. To record your results for sotabench, initialise
an ImageNet evaluator object to name the model (and optionally) link to a paper:

``` python
from sotabencheval.image_classification import ImageNetEvaluator

evaluator = ImageNetEvaluator(
             model_name='ResNeXt-101-32x8d',
             paper_arxiv_id='1611.05431')
```

For each batch of predictions made by your model, pass a dictionary of keys as image IDs and values as 
output predictions to the `evaluator.add` method:

``` python
evaluator.add(dict(zip(image_ids, batch_output)))
```
Then after you have accumulated all the predictions:

``` python
evaluator.save()
```

This will ensure results are evaluated and saved when they are run on the [sotabench](http://www.sotabench.com) server.

Below you can see a working `sotabench.py` file added to the [torchvision](https://github.com/pytorch/vision) repository 
to test one of its models, integrating the evaluation code from above:

``` python
import numpy as np
import PIL
import torch
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnext101_32x8d
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet

from sotabencheval.image_classification import ImageNetEvaluator
from sotabencheval.utils import is_server

if is_server():
    DATA_ROOT = './.data/vision/imagenet'
else: # local settings
    DATA_ROOT = '/home/ubuntu/my_data/'

model = resnext101_32x8d(pretrained=True)

input_transform = transforms.Compose([
    transforms.Resize(256, PIL.Image.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_dataset = ImageNet(
    DATA_ROOT,
    split="val",
    transform=input_transform,
    target_transform=None,
    download=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

model = model.cuda()
model.eval()

evaluator = ImageNetEvaluator(
                 model_name='ResNeXt-101-32x8d',
                 paper_arxiv_id='1611.05431')

def get_img_id(image_name):
    return image_name.split('/')[-1].replace('.JPEG', '')

with torch.no_grad():
    for i, (input, target) in enumerate(test_loader):
        input = input.to(device='cuda', non_blocking=True)
        target = target.to(device='cuda', non_blocking=True)
        output = model(input
        image_ids = [get_img_id(img[0]) for img in test_loader.dataset.imgs[i*test_loader.batch_size:(i+1)*test_loader.batch_size]]
        evaluator.add(dict(zip(image_ids, list(output.cpu().numpy()))))
    
evaluator.save()
```

**Step Two : Run locally to verify that it works** 

```
python sotabench.py
```

You can also run the logic in a Jupyter Notebook if that is your preferred workflow.

**Step Three : Login and connect your repository to [sotabench](http://www.sotabench.com)**

After you connect your repository, the sotabench servers will re-evaluate your model on every commit, 
to ensure the model is working and results are up-to-date - including if you add additional models to the benchmark file.

## Installation

The library requires Python 3.6+. You can install via pip:

```
pip install sotabencheval
```

## Support

If you get stuck you can head to our [Discourse](http://forum.sotabench.com) forum where you ask
questions on how to use the project. You can also find ideas for contributions,
and work with others on exciting projects.