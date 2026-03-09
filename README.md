# TransGeoCLIP

This is the code repository for the paper "When Vision Misleads, Let Location Speak: A Worldwide Image Geo-Localization Method via Location Attention Mechanism and Large Multimodal Models"

## Environment

```python
# Traning on CUDA Version: 12.8
conda create -n TransGeoCLIP python=3.10
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
```

## Data

For the IM2GPS, IM2GPS3k, YFCC4k, and YFCC26k datasets of the public test set, you can refer to the following links to query:
http://www.mediafire.com/

In the data directory, the metadata of the IM2GPS, IM2GPS3k, YFCC4k, and YFCC26k datasets are stored.

For the training dataset MP16-Pro dataset, you can visit the following link to query:

https://huggingface.co/datasets/Jia-py/MP16-Pro/tree/main

In addition, you can find the original image of the TwinBuilds dataset in the data directory, which you can randomly crop to create your own TwinBuilds dataset.  TwinBuilds raw data contains three sets of famous landmarks widely distributed in 14 different cities around the world, for each landmark we took three samples in the same city. We equipped each landmark with location information about its location.

## Running samples

1.Training model

2.Building index

3.Initial retrieval

4.lmms retrieval