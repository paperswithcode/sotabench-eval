# PASCAL VOC 2012

![VOC Dataset Examples](img/pascalvoc2012.png)

You can view the PASCAL VOC 2012 leaderboard [here](https://sotabench.com/benchmarks/semantic-segmentation-on-pascal-voc-2012).

## Getting Started

You'll need the following in the root of your repository:

- `sotabench.py` file - contains benchmarking logic; the server will run this on each commit
- `requirements.txt` file - Python dependencies to be installed before running `sotabench.py`
- `sotabench_setup.sh` *(optional)* - any advanced dependencies or setup, e.g. compilation

You can write whatever you want in your `sotabench.py` file to get model predictions on the VOC 2012 dataset. For example,
PyTorch users might use torchvision to load the dataset.

But you will need to record your results for the server, and you'll want to avoid doing things like
downloading the dataset on the server. So you should:

- **Point to the server VOC 2012 data paths** - popular datasets are pre-downloaded on the server.
- **Include an Evaluation object** in `sotabench.py` file to record the results.
- **Use Caching** *(optional)* - to speed up evaluation by hashing the first batch of predictions.
 
We explain how to do these various steps below.
 
## Server Data Location 

The VOC 2012 data is located in the root of your repository on the server at `.data/vision/voc2012`. In this folder is contained:

- `VOCtrainval_11-May-2012.tar` - containing validation images and annotations

Your local VOC 2012 files may have a different file directory structure, so you
can use control flow like below to change the data path if the script is being
run on sotabench servers:

``` python
from sotabencheval.utils import is_server

if is_server():
    DATA_ROOT = './.data/vision/voc2012'
else: # local settings
    DATA_ROOT = '/home/ubuntu/my_data/'
```

This will detect if `sotabench.py` is being run on the server and change behaviour accordingly.

## How Do I Initialize an Evaluator?

Add this to your code - before you start batching over the dataset and making predictions:

``` python
from sotabencheval.semantic_segmentation import PASCALVOCEvaluator

evaluator = PASCALVOCEvaluator(model_name='My Super Model')
```  
       
If you are reproducing a model from a paper, then you can enter the arXiv ID. If you
put in the same model name string as on the [leaderboard](https://sotabench.com/benchmarks/semantic-segmentation-on-pascal-voc-2012)
then you will enable direct comparison with the paper. For example:

``` python
from sotabencheval.semantic_segmentation import PASCALVOCEvaluator

evaluator = PASCALVOCEvaluator(model_name='PSPNet', paper_arxiv_id='1612.01105')
``` 

The above will directly compare with the result of the paper when run on the server.

## How Do I Evaluate Predictions?

The evaluator object has an `.add()` method to submit predictions by batch or in full.

For PASCAL there are two required arguments: `outputs`, a 1D np.ndarray of semantic class predictions per label, 
and `targets`, a 1D np.ndarray of ground truth semantic classes per pixel. In other words, it requires flattened
inputs and outputs.

To elaborate, suppose you are making predictions, batch by batch, and have your model output 
and the original targets with batch_size `32`, and image size `(520, 480)`. The shape of your outputs might look like:

``` python
batch_output.shape
>> (32, 21, 520, 480) # where 21 is the number of VOC classes

batch_target.shape
>> (32, 520, 480)
```

We can flatten the entire output and targets to 1D vectors for each pixel:

``` python
flattened_batch_output.shape
>> (7987200) # flatten by taking the max class prediction
             #  (batch_output.argmax(1).flatten() in torch with class as second dimension)

flattened_batch_target.shape
>> (7987200) # (batch_target.flatten() in torch)
```

The output might look something like this:

``` python
flattened_batch_output
>> array([6, 6, 6, 6, 6, ...])

flattened_batch_target
>> array([6, 6, 6, 6, 6, ...])
```

In both cases, the prediction and ground truth have class 6 as the semantic label for the first 5
pixels - so the model is correct.

These flattened arrays can then be passed into the .add() method of the evaluator

``` python
my_evaluator.update(outputs=flattened_batch_output,
                            targets=flattened_batch_target)
```

You can do this all at once in a single call to `add()`, but more naturally, you will 
probably loop over the dataset and call the method for the outputs of each batch.
That would like something like this (for a PyTorch example):

``` python
evaluator = PASCALVOCEvaluator(root=DATA_ROOT, dataset_year='2012', split='val', model_name='FCN (ResNet-101)',
                              paper_arxiv_id='1605.06211')

with torch.no_grad():
    for image, target in tqdm.tqdm(data_loader_test):
        image, target = image.to('cuda'), target.to('cuda')
        output = model(image)
        output = output['out']
        
        evaluator.add(output.argmax(1).flatten().cpu().numpy(), target.flatten().cpu().numpy())
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
evaluator = PASCALVOCEvaluator(root='./data', dataset_year='2012', split='val', model_name='FCN (ResNet-101)',
                              paper_arxiv_id='1605.06211')

with torch.no_grad():
    for image, target in tqdm.tqdm(data_loader_test):
        image, target = image.to('cuda'), target.to('cuda')
        output = model(image)
        output = output['out']

        evaluator.add(output.argmax(1).flatten().cpu().numpy(), target.flatten().cpu().numpy())
        if evaluator.cache_exists:
            break

evaluator.save()
```

If the hash is the same as in the server, we infer that the model hasn't changed, so
we simply return hashed results rather than running the whole evaluation again.

Caching is very useful if you have large models, or a repository that is evaluating
multiple models, as it speeds up evaluation significantly.
    
## A full sotabench.py example

Below we show an implementation for a model from the torchvision repository. This
incorporates all the features explained above: (a) using the server data root, 
(b) using the ImageNet Evaluator, and (c) caching the evaluation logic:

``` python
import PIL
import torch
import torchvision
from torchvision.models.segmentation import fcn_resnet101
import torchvision.transforms as transforms
import tqdm

from sotabench_transforms import Normalize, Compose, Resize, ToTensor

from sotabencheval.semantic_segmentation import PASCALVOCEvaluator
from sotabencheval.utils import is_server

if is_server():
    DATA_ROOT = './.data/vision/voc2012'
else: # local settings
    DATA_ROOT = '/home/ubuntu/my_data/'

MODEL_NAME = 'fcn_resnet101'

def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)
    return batched_imgs

def collate_fn(batch):
    images, targets = list(zip(*batch))
    batched_imgs = cat_list(images, fill_value=0)
    batched_targets = cat_list(targets, fill_value=255)
    return batched_imgs, batched_targets

device = torch.device('cuda')

normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
my_transforms = Compose([Resize((520, 480)), ToTensor(), normalize])

dataset_test = torchvision.datasets.VOCSegmentation(root=DATA_ROOT, year='2012', image_set="val", 
                                                    transforms=my_transforms, download=True)
test_sampler = torch.utils.data.SequentialSampler(dataset_test)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=32,
    sampler=test_sampler, num_workers=4,
    collate_fn=collate_fn)

model = torchvision.models.segmentation.__dict__['fcn_resnet101'](num_classes=21, pretrained=True)
model.to(device)
model.eval()

evaluator = PASCALVOCEvaluator(root=DATA_ROOT, dataset_year='2012', split='val', model_name='FCN (ResNet-101)',
                              paper_arxiv_id='1605.06211')

with torch.no_grad():
    for image, target in tqdm.tqdm(data_loader_test):
        image, target = image.to('cuda'), target.to('cuda')
        output = model(image)
        output = output['out']
        
        evaluator.add(output.argmax(1).flatten().cpu().numpy(), target.flatten().cpu().numpy())
        if evaluator.cache_exists:
            break
        
evaluator.save()
```

## Need More Help?

Head on over to the [Computer Vision](https://forum.sotabench.com/c/cv) section of the sotabench
forums if you have any questions or difficulties.
