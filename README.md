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

# Autoencoder architecture

![image](https://github.com/olley102/PopulationCountClustering/blob/main/autoencoder_segmentation.png?raw=true)

The autoencoder above is used for finding latent representations of individual pixels. A sliding window of size 7 × 7 is considered the input to the network. At each epoch, a random sample of pixels is chosen as the central pixels of the windows. The optical flow velocities and positions of the pixels are concatenated to the image to form windows of size 7 × 7 × 5. The reconstruction is of size 7 × 7 × 3, with position discarded. With all information discarded except the central pixel, the reconstruction and encoding of each pixel is embedded into a 2D grid of the same shape as the input.

# Matching clusters across frames

The tracking of the image segmentation across frames is achieved by comparing each cluster pair in consecutive frames and finding the best fit for cosine similarity of optical flow vectors. We define inter-layer connections by the relation $\alpha \sim \beta \iff (\alpha \rightarrow \beta)\lor(\alpha \leftarrow \beta)$ where

$$\alpha \rightarrow \beta \iff \alpha = \max_{c\in C^{(1)}}\left[ \sum_{i\in\beta\cap c} \frac{\left< v_i^{(1)}, w_i^{(2)} \right>}{\lVert v_i^{(1)}\rVert\lVert w_i^{(2)}\rVert} \leq | \beta \cap c | \right],$$

where $\beta\in C^{(2)}$, $C^{(k)}$ is the partitioning of frame $k$, and $v_i^{(1)}$ are the _forwards_ optical flow vectors for frame 1 and $w_i^{(2)}$ are the _backwards_ optical flow vectors for frame 2. The backwards relation $\alpha \leftarrow \beta$ is defined similarly, so that for each $\alpha$ we find a $\beta$ by a maximization task. The relations $\rightarrow$ and $\leftarrow$ account for cases where clusters in frame 1 either grow, shrink, merge or split into clusters in frame 2.
