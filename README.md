# Population count clustering

Image segmentation is the process of dividing an image into subsets of pixels based on a partitioning scheme (hard assignment) or probability distribution (soft assignment). In this project, we consider the task of image segmentation for identifying communities of population in population count raster data from Google Earth Engine. Furthermore, we devise a method of inferring inter-layer connections between partitions in consecutive frames. We outline a pipeline for image segmentation and tracking as follows:
- Determining the velocities of features in an image via optical flow.
- Discovering latent features of pixels via an autoencoder neural network with reconstruction MSE of 2e-4 (±7e-5) on normalised windows of size 7 × 7 × 3.
- Clustering latent features via KMeans and DBSCAN.
- Inferring inter-layer connections between partitions in consecutive frames via cosine similarity of optical flow vectors.

# In this repository

## Packages

`image_processing`: Python package used throughout project

- `analysis.py`: used for analysing performance of clustering and encoding algorithms
- `nn.py`: holds the `WindowAE` neural network class
- `optical_flow.py`: holds the **optical flow** algorithm

## Scripts

- `GPWv411_popcount_filtering.js`: JavaScript used in a Google Earth Engine project to download image data
- `automate_project.py`: reduce code duplication for notebooks

## Notebooks

- `load_gpw_rectangles.ipynb`: load image data saved from Google Earth Engine with `rasterio` package
- `optical_flow.ipynb`: experimentation with optical flow algorithm and numerical derivatives
- `image_processing_test.ipynb`: test `image_processing` for optical flow
- `gpw_optical_flow.ipynb`: apply optical flow to image data
- `autoencoder_image_segmentation.ipynb`: apply **autoencoder** to image data and apply **KMeans** clustering to encodings
- `clustering_image_segmentation.ipynb`: apply **DBSCAN** clustering to result from KMeans to produce highly complex partitions
- `object_tracking.ipynb`: apply **cosine similarity** rule to match clusters across consecutive frames
- `project_automation.ipynb`: automate project using `automate_project.py` to reduce code duplication
- `performance_analysis.ipynb`: graphically analyse the performance of various methods used in project
