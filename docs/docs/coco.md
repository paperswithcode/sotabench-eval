# COCO

![COCO Dataset Examples](img/coco.jpg)

You can view the COCO minival leaderboard [here](https://sotabench.com/benchmarks/object-detection-on-coco-minival).

## Getting Started

You'll need the following in the root of your repository:

- `sotabench.py` file - contains benchmarking logic; the server will run this on each commit
- `requirements.txt` file - Python dependencies to be installed before running `sotabench.py`
- `sotabench_setup.sh` *(optional)* - any advanced dependencies or setup, e.g. compilation

You can write whatever you want in your `sotabench.py` file to get model predictions on the COCO dataset. For example,
PyTorch users might use torchvision to load the dataset.

But you will need to record your results for the server, and you'll want to avoid doing things like
downloading the dataset on the server. So you should:

- **Point to the server COCO data paths** - popular datasets are pre-downloaded on the server.
- **Include an Evaluation object** in `sotabench.py` file to record the results.
- **Use Caching** *(optional)* - to speed up evaluation by hashing the first batch of predictions.
 
We explain how to do these various steps below.
 
## Server Data Location 

The COCO validation data is located in the root of your repository on the server at `.data/vision/coco`. In this folder is contained:

- `annotations_trainval2017.zip` - containing annotations for the validation images
- `val2017.zip` - containing the validation images

Your local COCO files may have a different file directory structure, so you
can use control flow like below to change the data path if the script is being
run on sotabench servers:

``` python
from sotabencheval.utils import is_server

if is_server():
    DATA_ROOT = './.data/vision/coco'
else: # local settings
    DATA_ROOT = '/home/ubuntu/my_data/'
```

This will detect if `sotabench.py` is being run on the server and change behaviour accordingly.

## How Do I Initialize an Evaluator?

Add this to your code - before you start batching over the dataset and making predictions:

``` python
from sotabencheval.object_detection import COCOEvaluator

evaluator = COCOEvaluator(model_name='My Super Model')
```
         
If you are reproducing a model from a paper, then you can enter the arXiv ID. If you
put in the same model name string as on the [leaderboard](https://sotabench.com/benchmarks/object-detection-on-coco-minival)
then you will enable direct comparison with the paper's model. For example:

``` python
from sotabencheval.object_detection import COCOEvaluator

evaluator = COCOEvaluator(model_name='Mask R-CNN', paper_arxiv_id='1703.06870')
```

The above will directly compare with the result of the paper when run on the server.

## How Do I Evaluate Predictions?

The evaluator object has an [.add()](https://github.com/paperswithcode/sotabench-eval/blob/a788d17252913e5f2d24733845de80aec23101fb/sotabencheval/object_detection/coco.py#L187) method to submit predictions by batch or in full.

For COCO the expected input is a list of dictionaries, where each dictionary contains detection information
that will be used by the [loadRes](https://github.com/paperswithcode/sotabench-eval/blob/a788d17252913e5f2d24733845de80aec23101fb/sotabencheval/object_detection/coco_eval.py#L236) method based on the [pycocotools](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools) API. 

Each detection can take a dictionary
like the following:

``` python
{'image_id': 397133, 'bbox': [386.1628112792969, 69.48855590820312, 110.14895629882812, 278.2847595214844],
'score': 0.999152421951294, 'category_id': 1}
```

For this benchmark, only bounding box detection ('bbox') is performed at present.

You can do this all at once in a single call to `add()`, but more naturally, you will 
probably loop over the dataset and call the method for the outputs of each batch.
That would look something like this (for a PyTorch example):

``` python
...

evaluator = COCOEvaluator(
                 model_name='Mask R-CNN',
                 paper_arxiv_id='1703.06870')

with torch.no_grad():
    for i, (input, target) in enumerate(data_loader):
        ...
        output = model(input)
        # potentially formatting of the output here to be a list of dicts
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

Below we show an implementation for a model from the torchvision repository. This
incorporates all the features explained above: (a) using the server data root, 
(b) using the COCO Evaluator, and (c) caching the evaluation logic. Note that the
torchbench dependency is just to get some processing logic and transforms; the evaluation
is done with sotabencheval.

``` python
import os
import tqdm 
import torch
from torch.utils.data import DataLoader
from torchbench.utils import send_model_to_device
from torchbench.object_detection.transforms import Compose, ConvertCocoPolysToMask, ToTensor
import torchvision
import PIL

from sotabencheval.object_detection import COCOEvaluator
from sotabencheval.utils import is_server

if is_server():
    DATA_ROOT = './.data/vision/coco'
else: # local settings
    DATA_ROOT = '/home/ubuntu/my_data/'

def coco_data_to_device(input, target, device: str = "cuda", non_blocking: bool = True):
    input = list(inp.to(device=device, non_blocking=non_blocking) for inp in input)
    target = [{k: v.to(device=device, non_blocking=non_blocking) for k, v in t.items()} for t in target]
    return input, target

def coco_collate_fn(batch):
    return tuple(zip(*batch))

def coco_output_transform(output, target):
    output = [{k: v.to("cpu") for k, v in t.items()} for t in output]
    return output, target

transforms = Compose([ConvertCocoPolysToMask(), ToTensor()])
   
model = torchvision.models.detection.__dict__['maskrcnn_resnet50_fpn'](num_classes=91, pretrained=True)

model, device = send_model_to_device(
    model, device='cuda', num_gpu=1
)
model.eval()

model_output_transform = coco_output_transform
send_data_to_device = coco_data_to_device
collate_fn = coco_collate_fn

test_dataset = torchbench.datasets.CocoDetection(
    root=os.path.join(DATA_ROOT, "val%s" % '2017'),
    annFile=os.path.join(
        DATA_ROOT, "annotations/instances_val%s.json" % '2017'
    ),
    transform=None,
    target_transform=None,
    transforms=transforms,
    download=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_fn,
)
test_loader.no_classes = 91  # Number of classes for COCO Detection

iterator = tqdm.tqdm(test_loader, desc="Evaluation", mininterval=5)

evaluator = COCOEvaluator(
    root=DATA_ROOT,
    model_name='Mask R-CNN (ResNet-50-FPN)',
    paper_arxiv_id='1703.06870'

def prepare_for_coco_detection(predictions):
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        boxes = prediction["boxes"]
        boxes = convert_to_xywh(boxes).tolist()
        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results

def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

with torch.no_grad():
    for i, (input, target) in enumerate(iterator):
        input, target = send_data_to_device(input, target, device=device)
        original_output = model(input)
        output, target = model_output_transform(original_output, target)
        result = {
            tar["image_id"].item(): out for tar, out in zip(target, output)
        }
        result = prepare_for_coco_detection(result)

        evaluator.update(result)
        
        if evaluator.cache_exists:
            break

evaluator.save()
```

## Need More Help?

Head on over to the [Computer Vision](https://forum.sotabench.com/c/cv) section of the sotabench
forums if you have any questions or difficulties.
