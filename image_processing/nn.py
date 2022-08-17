import os
import math
import numpy as np
from keras import Input, Model
from keras.layers import Activation, Flatten, Dense, Reshape
from keras.callbacks import ModelCheckpoint


class WindowAE:
    """
    Sliding window autoencoder for encoding local information for an image.
    """
    def __init__(self, window_size=(7, 7), num_channels=1, encoder_sizes=None,
                 decoder_sizes=None):
        self.model = None
        self.encoder = None
        self._built = False
        self.max = 1.0
        self.min = 0.0
        self.window_size = window_size
        self.num_channels = num_channels
        self.encoder_sizes = encoder_sizes
        self.decoder_sizes = decoder_sizes
        self.callbacks = []

    def auto_decoder_sizes(self, encoder_sizes):
        self.encoder_sizes = encoder_sizes
        flat_size = self.window_size[0] * self.window_size[1] * self.num_channels
        self.decoder_sizes = (*encoder_sizes[-2::-1], flat_size)

    def make(self):
        if (
                not self._built and
                self.encoder_sizes is not None and
                self.decoder_sizes is not None
        ):
            stack_size = self.window_size + (self.num_channels+2,)
            input_window = Input(stack_size)

            x = Flatten()(input_window)
            for s in self.encoder_sizes:
                x = Dense(s, activation='relu')(x)

            encoded = Activation('linear')(x)

            for s in self.decoder_sizes:
                x = Dense(s, activation='relu')(x)

            decoded = Reshape(stack_size)(x)  # must match unraveled decoder_sizes[-1]

            self.model = Model(input_window, decoded)
            self.encoder = Model(input_window, encoded)

    def compile(self):
        self.model.compile(loss='mse', optimizer='adam', metrics=['mse'])

    def fit_transform(self, x):
        self.max = np.max(x)
        self.min = np.min(x)

    def transform_x(self, x):
        # Normalize.
        x_norm = (x - self.min) / (self.max - self.min)

        # Zero padding.
        pad = (*(math.floor(s/2) for s in self.window_size), 0)
        pad_widths = tuple((p, p) for p in pad)
        x_pad = np.pad(x_norm, pad_widths)

        # Stack array with positional information.
        pos_i = np.arange(-pad[0], x.shape[0]+pad[0], dtype='float')
        pos_j = np.arange(-pad[1], x.shape[1]+pad[1], dtype='float')
        x_i = np.outer(pos_i, np.ones(x_pad.shape[1], dtype='float'))
        x_j = np.outer(np.ones(x_pad.shape[0], dtype='float'), pos_j)
        x_full = np.dstack((x_pad, x_i, x_j))

        return x_full

    def transform_y(self, y):
        return self.transform_x(y)

    def encode(self, x):
        x_full = self.transform_x(x)
        enc_full = np.zeros((*x.shape[:2], self.encoder_sizes[-1]))

        for p in range(x.shape[0]*x.shape[1]):
            unravel_p = np.unravel_index(p, x.shape[:2])
            window = x_full[unravel_p[0]:unravel_p[0]+self.window_size[0],
                            unravel_p[1]:unravel_p[1]+self.window_size[1]]
            enc_p = self.encoder.predict(window)
            enc_full[unravel_p] = np.flatten(enc_p)

        return enc_full

    def predict(self, x):
        x_full = self.transform_x(x)
        pred = np.zeros_like(x)
        pad_half = tuple(math.floor(s/2) for s in self.window_size)

        for p in range(x.shape[0]*x.shape[1]):
            unravel_p = np.unravel_index(p, x.shape[:2])
            window = x_full[unravel_p[0]:unravel_p[0]+self.window_size[0],
                            unravel_p[1]:unravel_p[1]+self.window_size[1]]
            window = np.expand_dims(window, axis=0)
            pred_window = self.model.predict(window)

            # Store central pixel of pred_window in pred.
            pred[unravel_p] = pred_window[unravel_p[0]+pad_half[0],
                                          unravel_p[1]+pad_half[1]]

        return pred

    def fit(self, x, y, epochs=1, batch_size=32, **kwargs):
        x_full = self.transform_x(x)
        y_full = self.transform_y(y)

        history = []

        for ep in range(epochs):
            # Make a random choice of pixels.
            ravel_choice = np.random.choice(np.arange(x.shape[0]*x.shape[1]),
                                            batch_size, replace=False)
            unravel_choice = np.column_stack(np.unravel_index(ravel_choice,
                                                              x.shape[:2]))

            # Make windows for each pixel and stack them.
            # Central pixel is chosen pixel, but we don't have to shift start
            # indices.
            x_stack = np.stack(tuple(x_full[i:i+self.window_size[0],
                                     j:j+self.window_size[1]]
                                     for i, j in unravel_choice), axis=0)
            y_stack = np.stack(tuple(y_full[i:i+self.window_size[0],
                                     j:j+self.window_size[1]]
                                     for i, j in unravel_choice), axis=0)

            # Train network on stack.
            history.append(self.model.fit(x_stack, y_stack, epochs=ep+1,
                                          initial_epoch=ep, callbacks=self.callbacks,
                                          **kwargs))

        return history

    def make_callback(self, filepath, save_weights_only=True,
                      save_best_only=False, **kwargs):
        checkpoint_callback = ModelCheckpoint(filepath=filepath,
                                              save_weights_only=save_weights_only,
                                              save_best_only=save_best_only,
                                              **kwargs)
        self.callbacks.append(checkpoint_callback)

    def clear_callbacks(self):
        self.callbacks = []

    def load_epoch(self, filepath, epoch):
        assert isinstance(epoch, int)
        try:
            fp = filepath.format(epoch=epoch)  # fill {epoch} keyword
            # Check file exists and load weights.
            assert os.path.isfile(fp)
            self.model.load_weights(filepath.format(epoch=epoch))
            return True
        except Exception as e:
            if hasattr(e, 'message'):
                print(e.message)
            else:
                print(e)
            return False
