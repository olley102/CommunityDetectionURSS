import rasterio
import numpy as np
import image_processing as ip
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler


class Project:
    def __init__(self, data_fp, checkpoint_fp):
        self.data_fp = data_fp
        self.checkpoint_fp = checkpoint_fp
        self.autoencoders = []
        self.data = None
        self.images = None
        self.encodings = []
        self.kmeans_images = None
        self.dbscan_images = None
        self.scaler = StandardScaler()
        self.kmeans_kwargs = dict(init='random', n_init=10, max_iter=10, random_state=42)

    def load_data(self, num_frames):
        with rasterio.open(self.data_fp.format(0), 'r') as ds:
            time0 = ds.read()

        self.data = np.zeros((time0.shape[1], time0.shape[2], num_frames), dtype='float')
        self.data[..., 0] = time0[0]

        for i in range(num_frames-1):
            with rasterio.open(self.data_fp.format(i + 1), 'r') as ds:
                self.data[..., i + 1] = ds.read()[0]

        self.images = np.zeros_like(self.data)
        self.images[~np.isnan(self.data)] += self.data[~np.isnan(self.data)]

        self.kmeans_images = np.zeros_like(self.data)
        self.dbscan_images = -np.ones_like(self.data)

    def fit_autoencoders(self, num_frames):
        uv = ip.optical_flow.iteration(self.images, 1, alpha=10, use_previous=True, centering=(0, 0, 0))

        for i in range(num_frames):
            image = np.dstack((self.images[..., i], np.moveaxis(uv[..., i], 0, -1)))
            image = np.clip(image, None, 1000)
            ae = ip.nn.WindowAE(window_size=(7, 7), num_channels=3)
            ae.auto_decoder_sizes((128, 64, 16))
            ae.make()
            ae.compile()
            ae.make_callback(self.checkpoint_fp.format(frame=i, epoch='epoch'), period=10)
            ae.fit_transform(image)
            history = ae.fit(image, image, epochs=200, batch_size=1000)
            self.autoencoders.append(ae)

            yield history

    def load_epochs(self, *epochs):
        for i in range(len(epochs)):
            self.autoencoders[i].load_epoch(self.checkpoint_fp.format(frame=i, epoch='epoch'), epochs[i])

    def encode(self, num_frames):
        batch_size = self.images.shape[1] * 10
        for i in range(num_frames):
            encoding = self.autoencoders[i].encode(
                self.images[..., i], verbose=True, batch_size=batch_size
            ).reshape(-1, 16)
            self.encodings.append(encoding)
            yield encoding

    def predict(self, num_frames):
        batch_size = self.images.shape[1] * 10
        for i in range(num_frames):
            yield self.autoencoders[i].predict(self.images[..., i], verbose=True, batch_size=batch_size)

    def sse_search(self, num_frames):
        for i in range(num_frames):
            scaled_features = self.scaler.fit_transform(self.encodings[i])
            sse = []

            for k in range(2, 21):
                print(f'Trying {k} clusters for frame {i}')
                kmeans = KMeans(n_clusters=k, **self.kmeans_kwargs)
                kmeans.fit(scaled_features)
                sse.append(kmeans.inertia_)

            yield sse

    def kmeans_segmentation(self, num_frames, n_clusters=10):
        for i in range(num_frames):
            scaled_features = self.scaler.fit_transform(self.encodings[i])
            kmeans = KMeans(n_clusters=n_clusters, **self.kmeans_kwargs)
            kmeans.fit(scaled_features)
            kmeans_image = kmeans.labels_.reshape(self.data.shape[:2])
            self.kmeans_images[..., i] = kmeans_image

    def dbscan_segmentation(self, num_frames, n_clusters, eps=10, min_samples=50):
        for i in range(num_frames):
            label_pos = [np.array(np.where(self.kmeans_images[i] == k)).T for k in range(10)]

            for k in range(n_clusters):
                clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(label_pos[k])
                img = np.zeros_like(self.kmeans_images[i])
                img[self.kmeans_images[i] == k] = clustering.labels_ + 1
                pos_img = img[img > 0]
                self.dbscan_images[img > 0, i] += pos_img + self.dbscan_images[img > 0, i].max() + 1

        self.dbscan_images[np.isnan(self.data)] = -1

    def object_tracking(self, num_frames):
        for i in range(num_frames-1):
            yield ip.optical_flow.object_tracking(
                self.images[..., i], self.images[..., i+1],
                self.dbscan_images[..., i], self.dbscan_images[..., i+1],
                n=1, alpha=10
            )
