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
    def __init__(self, window_size=(7, 7), num_channels=1, encoder_sizes=None, decoder_sizes=None,
                 bottleneck_activation='relu', final_activation='linear', encoding_regularizer=None):
        self.model = None
        self.encoder = None
        self._built = False
        self.max = np.ones(num_channels)
        self.min = np.zeros(num_channels)
        self.window_size = window_size
        self.num_channels = num_channels
        self.encoder_sizes = encoder_sizes
        self.decoder_sizes = decoder_sizes
        self.bottleneck_activation = bottleneck_activation
        self.final_activation = final_activation
        self.encoding_regularizer = encoding_regularizer
        self.callbacks = []

    def auto_decoder_sizes(self, encoder_sizes):
        self.encoder_sizes = encoder_sizes
        flat_size = self.window_size[0] * self.window_size[1] * self.num_channels
        self.decoder_sizes = (*encoder_sizes[-2::-1], flat_size)

    def make(self):
        if not self._built and self.encoder_sizes is not None and self.decoder_sizes is not None:
            stack_size = self.window_size + (self.num_channels+2,)
            input_window = Input(stack_size)

            x = Flatten()(input_window)
            for s in self.encoder_sizes[:-1]:
                x = Dense(s, activation='relu', activity_regularizer=self.encoding_regularizer)(x)

            encoded = Dense(self.encoder_sizes[-1], activation=self.bottleneck_activation,
                            kernel_regularizer=self.encoding_regularizer)(x)
            x = Activation('linear')(encoded)

            for s in self.decoder_sizes[:-1]:
                x = Dense(s, activation='relu')(x)

            x = Dense(self.decoder_sizes[-1], activation=self.final_activation)(x)
            # Note, decoder_sizes[-1] does not have to be equal to stack_size in last dimension
            decoded = Reshape((*stack_size[:-1], -1))(x)  # must match unraveled decoder_sizes[-1]

            self.model = Model(input_window, decoded)
            self.encoder = Model(input_window, encoded)

    def compile(self):
        self.model.compile(loss='mse', optimizer='adam', metrics=['mse'])

    def fit_transform(self, x):
        self.max = np.max(x, axis=(0, 1))
        self.min = np.min(x, axis=(0, 1))

    def transform_x(self, x, concat_position=True):
        # Normalize.
        max_expand = np.expand_dims(self.max, axis=(0, 1))
        min_expand = np.expand_dims(self.min, axis=(0, 1))
        x_norm = (x - min_expand) / (max_expand - min_expand)

        # Zero padding.
        pad = (*(math.floor(s/2) for s in self.window_size), 0)
        pad_widths = tuple((p, p) for p in pad)
        x_pad = np.pad(x_norm, pad_widths)

        if concat_position:
            # Stack array with positional information.
            pos_i = np.arange(-pad[0], x.shape[0]+pad[0], dtype='float')
            pos_j = np.arange(-pad[1], x.shape[1]+pad[1], dtype='float')
            x_i = np.outer(pos_i, np.ones(x_pad.shape[1], dtype='float'))
            x_j = np.outer(np.ones(x_pad.shape[0], dtype='float'), pos_j)
            x_full = np.dstack((x_pad, x_i, x_j))
            return x_full
        else:
            return x_pad

    def transform_y(self, y, concat_position=False):
        return self.transform_x(y, concat_position=concat_position)

    def encode(self, x, verbose=False, batch_size=None):
        if batch_size is None:
            batch_size = x.shape[1]

        x_full = self.transform_x(x)
        enc_full = np.zeros((*x.shape[:2], self.encoder_sizes[-1]))

        pixel_count = 0
        while pixel_count < x.shape[0]*x.shape[1]:
            end_pixel = min(pixel_count + batch_size, x.shape[0]*x.shape[1])
            if verbose:
                print(f'Encoding pixels {pixel_count}:{end_pixel}')

            ravel_choice = np.arange(pixel_count, end_pixel)

            unravel_choice = np.unravel_index(ravel_choice, x.shape[:2])
            x_stack = np.stack(tuple(x_full[i:i+self.window_size[0], j:j+self.window_size[1]]
                                     for i, j in np.column_stack(unravel_choice)), axis=0)
            p_stack = self.encoder.predict(x_stack)
            enc_full[unravel_choice] = p_stack

            pixel_count = end_pixel

        # for p in range(x.shape[0]*x.shape[1]):
        #     unravel_p = np.unravel_index(p, x.shape[:2])
        #     if verbose:
        #         print(f'Encoding pixel {unravel_p}')
        #     window = x_full[unravel_p[0]:unravel_p[0]+self.window_size[0],
        #                     unravel_p[1]:unravel_p[1]+self.window_size[1]]
        #     window = np.expand_dims(window, axis=0)
        #     enc_p = self.encoder.predict(window)
        #     enc_full[unravel_p] = enc_p.flatten()

        # for i in range(x.shape[0]):
        #     if verbose:
        #         print(f'Predicting row {i}')
        #     x_stack = np.stack(tuple(x_full[i:i+self.window_size[0], j:j+self.window_size[1]]
        #                              for j in range(x.shape[1])), axis=0)
        #     enc_stack = self.encoder.predict(x_stack)
        #     enc_full[i] = enc_stack.reshape(-1, enc_full.shape[-1])

        return enc_full

    def predict(self, x, verbose=False, batch_size=None):
        if batch_size is None:
            batch_size = x.shape[1]

        x_full = self.transform_x(x)
        pred = np.zeros_like(x)
        pad_half = tuple(math.floor(s/2) for s in self.window_size)

        pixel_count = 0
        while pixel_count < x.shape[0]*x.shape[1]:
            end_pixel = min(pixel_count + batch_size, x.shape[0]*x.shape[1])
            if verbose:
                print(f'Predicting pixels {pixel_count}:{end_pixel}')

            ravel_choice = np.arange(pixel_count, end_pixel)

            unravel_choice = np.unravel_index(ravel_choice, x.shape[:2])
            x_stack = np.stack(tuple(x_full[i:i+self.window_size[0], j:j+self.window_size[1]]
                                     for i, j in np.column_stack(unravel_choice)), axis=0)
            p_stack = self.model.predict(x_stack)

            # print(f'p_stack.shape={p_stack.shape}')
            # print(f'np.any(p_stack)={np.any(p_stack)}')

            # Store central pixels of p_stack in pred. Remove predictions for positions.
            pred[unravel_choice] = p_stack[:, pad_half[0], pad_half[1], :self.num_channels]
            # print(f'pred[unravel_choice]={pred[unravel_choice]}')

            pixel_count = end_pixel

        # for i in range(x.shape[0]):
        #     if verbose:
        #         print(f'Predicting row {i}')
        #     x_stack = np.stack(tuple(x_full[i:i+self.window_size[0], j:j+self.window_size[1]]
        #                              for j in range(x.shape[1])), axis=0)
        #     p_stack = self.model.predict(x_stack)
        #
        #     # Store central pixel of p_stack in pred. Remove predictions for positions.
        #     pred[i] = p_stack[:, pad_half[0], pad_half[1], :-2]

        # for p in range(x.shape[0]*x.shape[1]):
        #     unravel_p = np.unravel_index(p, x.shape[:2])
        #     if verbose:
        #         print(f'Predicting pixel {unravel_p}')
        #     window = x_full[unravel_p[0]:unravel_p[0]+self.window_size[0],
        #                     unravel_p[1]:unravel_p[1]+self.window_size[1]]
        #     window = np.expand_dims(window, axis=0)
        #     pred_window = self.model.predict(window)
        #
        #     # Store central pixel of pred_window in pred. Remove predictions for positions.
        #     pred[unravel_p] = pred_window[0, pad_half[0], pad_half[1], :-2]

        # print(f'np.any(pred)={np.any(pred)}')

        # Reverse normalization.
        max_expand = np.expand_dims(self.max, axis=(0, 1))
        min_expand = np.expand_dims(self.min, axis=(0, 1))
        pred = pred * (max_expand - min_expand) + min_expand

        return pred

    def fit(self, x, y, epochs=1, batch_size=32, **kwargs):
        x_full = self.transform_x(x)
        y_full = self.transform_y(y)

        history = []

        for ep in range(epochs):
            # Make a random choice of pixels.
            ravel_choice = np.random.choice(np.arange(x.shape[0]*x.shape[1]), batch_size, replace=False)
            unravel_choice = np.column_stack(np.unravel_index(ravel_choice, x.shape[:2]))

            # Make windows for each pixel and stack them.
            # Central pixel is chosen pixel, but we don't have to shift start
            # indices.
            x_stack = np.stack(tuple(x_full[i:i+self.window_size[0], j:j+self.window_size[1]]
                                     for i, j in unravel_choice), axis=0)
            y_stack = np.stack(tuple(y_full[i:i+self.window_size[0], j:j+self.window_size[1]]
                                     for i, j in unravel_choice), axis=0)

            # Train network on stack.
            history.append(self.model.fit(x_stack, y_stack, epochs=ep+1, initial_epoch=ep, callbacks=self.callbacks,
                                          **kwargs))

        return history

    def make_callback(self, filepath, save_weights_only=True,
                      save_best_only=False, **kwargs):
        checkpoint_callback = ModelCheckpoint(filepath=filepath, save_weights_only=save_weights_only,
                                              save_best_only=save_best_only, **kwargs)
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
