# DELIUS

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

This is a PyTorch reproduction of the [original Keras implementation](https://github.com/gvessio/deep-clustering-art) of the [DELIUS algorithm](https://arxiv.org/pdf/2106.06234), which aims to produce effective clusters of visual arts data.

## Example

The following scatter plot displays a 2D visualization of the 10 clusters that were detected among the 10-dimensionals embeddings that were learned from the ~116k pictures in the [ArtGraph dataset](https://zenodo.org/records/6337958).

<img src="https://dagshub.com/GianmarcoTurchiano/DELIUS-Visual-Arts-Deep-Clustering/raw/341a79b8578bda2f43d1430553767e9e4351ed45/models/clustering_pics/clusters.png"/>

Here it follows a sampling of 5 pictures per cluster, which qualitatively highlights the similarities of pictures that ended up in the same cluster.

<img src="https://dagshub.com/GianmarcoTurchiano/DELIUS-Visual-Arts-Deep-Clustering/raw/341a79b8578bda2f43d1430553767e9e4351ed45/models/clustering_pics/samples.png"/>

--------

## Usage

Install the present module alongside its dependencies with the following command.

```
pip install git+https://github.com/GianmarcoTurchiano/DELIUS-Visual-Arts-Deep-Clustering.git
```

Given a variable `pics_dir_path`, standing for the path of a directory containing pictures (files in the format of `.jpeg`, `.png`, etc), the module may be employed as displayed here. 

```python
from delius.features_extraction import extract_densenet_embeddings
from delius.modules.embeddings_dataset import EmbeddingsDataset
from delius.encoder_pretraining import pretrain_encoder
from delius.clusters_initialization import init_clusters
from delius.deep_embedded_clustering import deep_embedded_clustering

embeddings = extract_densenet_embeddings(pics_dir_path)
dataset = EmbeddingsDataset(embeddings)
encoder = pretrain_encoder(dataset)
centroids, assignments = init_clusters(dataset, encoder)
dec = deep_embedded_clustering(encoder, dataset, centroids, assignments)
```

A Python notebook which also shows how to display the results can be found [here](notebooks/example.ipynb).

--------
