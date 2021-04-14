import keras
from keras import backend as K


class CustomVariationalLayer(keras.layers.Layer):

    def __init__(self, z_log_var, z_mean):
        super().__init__()
        self.z_log_var = z_log_var
        self.z_mean = z_mean

    def vae_loss(self, x, z_decoded):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
        xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)
        kl_loss = -5e-4 * K.mean(
            1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        loss = self.vae_loss(x, z_decoded)
        self.add_loss(loss, inputs=inputs)

        return x

    def get_config(self):
        return super().get_config()
