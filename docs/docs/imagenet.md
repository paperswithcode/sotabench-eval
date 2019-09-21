# ImageNet

You can view the ImageNet leaderboard [here](https://sotabench.com/benchmarks/image-classification-on-imagenet).

## Getting Started

The core things you will need to do is add the following to your repository:

- sotabench.py file - contains your benchmarking logic and is what the server will run on each commit.
- requirements.txt file - contains Python dependencies that will be installed before running sotabench.py
- *(optional)* sotabench_setup.sh - any advanced dependencies, for example if you need to compile any custom functions

You can write whatever you want in your sotabench.py file to get model predictions on the ImageNet - for example,
PyTorch users might import torch and then use torchvision to load the dataset.

So that results can be recorded on the server, however, and that you don't have to do things like download the data, you should:

- Point to the server ImageNet data paths (popular deep learning datasets are pre-downloaded on the server).
- Include a sotabencheval Evaluation object in sotabench.py file to record the results.
- *(optional)* Use sotabencheval caching to speed up evaluation.
 
## Server Data Location 

The ImageNet validation data is located in the root of your repository on the server at  .data/vision/imagenet. In this folder is contained:

- ILSVRC2012_devkit_t12.tar.gz - containing metadata
- ILSVRC2012_img_val.tar - containing the validation images

Your local ImageNet files may have a different file directory structure, so you
can use control flow like below to change the data path if the script is being
run on sotabench servers:

    from sotabencheval.utils import is_server
    
    if is_server():
        DATA_ROOT = './.data/vision/imagenet'
    else: # local settings
        DATA_ROOT = '/home/ubuntu/my_data/'

## How Do I Initialize an Evaluator?

Add this to your code - before you start batching over the dataset and making predictions:

    from sotabencheval.image_classification import ImageNetEvaluator
    
    evaluator = ImageNetEvaluator(model_name='My Super Model')
                 
If you are reproducing a model from a paper, then you can enter the ArXiv ID. If you
put in the same model name string as on the [leaderboard](https://sotabench.com/benchmarks/image-classification-on-imagenet)
then you will enable direct comparison with the paper. For example:

    from sotabencheval.image_classification import ImageNetEvaluator
    
    evaluator = ImageNetEvaluator(model_name='FixResNeXt-101 32x48d',
    paper_arxiv_id='1906.06423')
    
The above will directly compare with the result of the paper when you submit.

## How do I Evaluate Predictions?

The evaluator object has an .add() method where you can submit predictions.

For ImageNet the expected input as a dictionary of outputs, where each key is an
image ID from ImageNet and each value is a list or numpy array of logits for that 
image ID. For example:

    my_evaluator.add({'ILSVRC2012_val_00000293': np.array([1.04243, ...]),
    'ILSVRC2012_val_00000294': np.array([-2.3677, ...])})
    
You can do this all at once in a single call to add(), but more naturally, you will 
probably loop over the dataset and call the method for the outputs of each batch.

When you are done, you can see the results locally by running:

    my_evaluator.get_results()
    
But for the server you want to save the results by calling:

    my_evaluator.save()
    
## How Do I Cache Evaluation?
    
Sotabench reruns your script on every commit. This is good because it acts like 
continuous integration in checking for bugs and changes, but can be annoying
if the model hasn't changed (and the same thing is evaluated each time). 
Fortunately sotabencheval has caching methods that you can use.

The idea is that after the first batch, we hash the model outputs and the
current metrics and this tells us if the model is the same (given the dataset).
You can include hashing within an evaluation loop like follows (in the following
example for a PyTorch repository):

    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            input = input.to(device='cuda', non_blocking=True)
            target = target.to(device='cuda', non_blocking=True)
            output = model(input)
    
            image_ids = [get_img_id(img[0]) for img in test_loader.dataset.imgs[i*test_loader.batch_size:(i+1)*test_loader.batch_size]]
    
            evaluator.add(dict(zip(image_ids, list(output.cpu().numpy()))))
               
            if evaluator.cache_exists:
                break

    evaluator.save()
 
If the hash is the same as in the server, we infer that the model hasn't changed, so
we simply return hashed results rather than running the whole evaluation again.

Caching is very useful if you have large models, or a repository that is evaluating
multiple models, as it speeds up evaluation significantly.
    
## A full sotabench.py example

Below we show an implementation for a model from the torchvision repository. This
incorporates all the features explained above: (a) using the server data root, 
(b) using the ImageNet Evaluator, and (c) caching the evaluation logic.


    import numpy as np
    import PIL
    import torch
    from torchvision.models.resnet import resnext101_32x8d
    import torchvision.transforms as transforms
    from torchvision.datasets import ImageNet
    from torch.utils.data import DataLoader
    
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
                     paper_model_name='ResNeXt-101-32x8d',
                     paper_arxiv_id='1611.05431')
    
    def get_img_id(image_name):
        return image_name.split('/')[-1].replace('.JPEG', '')
    
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            input = input.to(device='cuda', non_blocking=True)
            target = target.to(device='cuda', non_blocking=True)
            output = model(input)
    
            image_ids = [get_img_id(img[0]) for img in test_loader.dataset.imgs[i*test_loader.batch_size:(i+1)*test_loader.batch_size]]
    
            evaluator.add(dict(zip(image_ids, list(output.cpu().numpy()))))
        
    evaluator.save()

## Need more help?

Head on over to the [Computer Vision](https://forum.sotabench.com/c/cv) section of the sotabench
forums if you have any questions or difficulties.
