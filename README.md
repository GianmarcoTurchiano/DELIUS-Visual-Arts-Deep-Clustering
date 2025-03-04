# DELIUS

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

This is a PyTorch reproduction of the [original Keras implementation](https://github.com/gvessio/deep-clustering-art) of the [DELIUS algorithm](https://link.springer.com/article/10.1007/s11263-022-01664-y), which aims to produce effective clusters of visual arts data.

## Example

The following scatter plot displays a 2D visualization of the 10 clusters which were detected among the 10-dimensionals embeddings that were learned from the ~116k pictures in the [ArtGraph dataset](https://zenodo.org/records/6337958).

<img src="https://dagshub.com/GianmarcoTurchiano/DELIUS-Visual-Arts-Deep-Clustering/raw/341a79b8578bda2f43d1430553767e9e4351ed45/models/clustering_pics/clusters.png"/>

Here it follows a sampling of 5 pictures per cluster, which qualitatively highlights the similarities between the pictures that ended up in the same cluster.

<img src="https://dagshub.com/GianmarcoTurchiano/DELIUS-Visual-Arts-Deep-Clustering/raw/341a79b8578bda2f43d1430553767e9e4351ed45/models/clustering_pics/samples.png"/>

--------

## Usage

Install the present module alongside its dependencies with the following command.

```
pip install git+https://github.com/GianmarcoTurchiano/DELIUS-Visual-Arts-Deep-Clustering.git
```

Given a variable `pics_dir_path`, standing for the path of a directory containing pictures (files in the format of `.jpeg`, `.png`, etc), the module may be employed as displayed here.

```python
from delius.clustering.modules.features_extractor import DenseNetFeaturesExtractor
from delius.clustering.features_extraction import extract_features
from delius.clustering.modules.features_dataset import FeaturesDataset
from delius.clustering.encoder_pretraining import pretrain_encoder
from delius.clustering.clusters_initialization import initialize_clusters
from delius.clustering.dec_fitting import fit_dec

extractor = DenseNetFeaturesExtractor()
names, features = extract_features(extractor, pics_dir_path)
dataset = FeaturesDataset(names, features)
encoder = pretrain_encoder(dataset)
centroids, assignments = initialize_clusters(dataset, encoder)
dec = fit_dec(encoder, dataset, centroids, assignments)
```

A Python notebook which also shows how to display the results can be found [here](notebooks/example.ipynb). You can run it on Kaggle onto the CIFAR-10 dataset [here](https://www.kaggle.com/code/gianmarcoturchiano/delius-visual-arts-deep-clustering).