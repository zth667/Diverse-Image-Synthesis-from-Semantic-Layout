# Photographic Image Synthesis with Cascaded Refinement Networks

This is a Tensorflow implementation of cascaded refinement networks to synthesize photographic images from semantic layouts.

<img src="https://people.eecs.berkeley.edu/~ke.li/projects/imle/scene_layouts/vids/diverse18_large.gif"/>

## Setup

### Requirement
Required python version: Python 2.7

Required python libraries: Tensorflow (>=1.0) + Scipy + Numpy + Pillow.

Tested in Ubuntu + Intel i7 CPU + Nvidia Titan X (Pascal) with Cuda (>=8.0) and CuDNN (>=5.0). CPU mode should also work with minor changes.

### Quick Start (Testing)
1. Clone this repository.
2. Download the VGG19 pretrained model by running "python download_models.py".
3. Download the pretrained model from [here](https://drive.google.com/open?id=1zQzeEGB715jufm0-9MbzbWswTvdiTzyr) and extract at the root directory
3. Run "test.py" to synthesize images.
4. The synthesized images are saved in "gta_demo/result/"

### Training

#### Dataset
You can download the pre-processed dataset from [here](https://drive.google.com/open?id=1e63Hl6I9ToE0VNiyUgEXDXMUd17DQvtQ) or you can download it from the official [website](https://download.visinf.tu-darmstadt.de/data/from_games/) and run "python preprocess.py" to process the images.

#### Pretraining
We train our model based on the CRN pretrained model. The pre-processed model (extra channels are added) can be downloaded from [here](https://drive.google.com/open?id=1Sbjzs_0CeDIrTUIn4uE98izY0vroY84V). We will release the pre-processing script soon.

#### Rarity estimation
Once you have downloaded the dataset, you can generate the rarity mask (for loss rebalancing) and rarity bins (for dataset rebalancing) by running "python gen_dataset_weight.py" or you can download the pre-generated ones from [here](https://drive.google.com/open?id=1MFEVGevOOcGytkMYiYakHAt6BssAuQaO)

#### Run
Run "python train.py" to start training

## Question
If you have any question or request about the code and data, please email me at bryanzhang97@gmail.com.

## License
MIT License
