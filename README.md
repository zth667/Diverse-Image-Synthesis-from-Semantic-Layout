<img src='diverse18.gif' align="right" width=250>


# Diverse Image Synthesis from Semantic Layouts via Conditional IMLE
### [Project](https://people.eecs.berkeley.edu/~ke.li/projects/imle/scene_layouts/) | [Paper](https://arxiv.org/pdf/1811.12373.pdf) <br>
This is a Tensorflow implementation of our method to generate diverse images from semantic layouts. <br><br>
[Diverse Image Synthesis from Semantic Layouts via Conditional IMLE](https://people.eecs.berkeley.edu/~ke.li/projects/imle/scene_layouts/)  
 [Ke Li](https://people.eecs.berkeley.edu/~ke.li/)<sup>\*</sup>, [Tianhao Zhang](https://zth667.github.io/)<sup>\*</sup>, [Jitendra Malik](https://people.eecs.berkeley.edu/~malik/)<br>
 (* equal contribution, alphabetical order)<br>
 In ICCV 2019.

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
We train our model based on the CRN pretrained model. The pre-processed model (extra channels are added) can be downloaded from [here](https://drive.google.com/open?id=1Sbjzs_0CeDIrTUIn4uE98izY0vroY84V). You can also download the CRN pretrained model from their [project](https://github.com/CQFIO/PhotographicImageSynthesis) and then preprocess the model by running "python preprocess_crn_model.py".

#### Rarity estimation
Once you have downloaded the dataset, you can generate the rarity mask (for loss rebalancing) and rarity bins (for dataset rebalancing) by running "python gen_dataset_weight.py" or you can download the pre-generated ones from [here](https://drive.google.com/open?id=1MFEVGevOOcGytkMYiYakHAt6BssAuQaO)

#### Run
Run "python train.py" to start training

## Question
If you have any question or request about the code and data, please email me at bryanzhang97@gmail.com.

## Citation

If you find this useful for your research, please use the following.

```
@inproceedings{li2019diverse,
  title={Diverse Image Synthesis from Semantic Layouts via Conditional IMLE},
  author={Ke Li and Tianhao Zhang and Jitendra Malik},  
  booktitle={Proceedings of the IEEE international conference on computer vision},
  year={2019}
}
```

## Acknowledgments
This code borrows heavily from [Photographic Image Synthesis with Cascaded Refinement Networks](https://github.com/CQFIO/PhotographicImageSynthesis).

## License
MIT License
