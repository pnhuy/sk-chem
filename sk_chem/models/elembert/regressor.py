import tensorflow as tf
from tensorflow.keras import layers, Model

from sk_chem.models.elembert.base import Config, ElemBertEmbedding, create_elembert_model
from sk_chem.models.elembert.tokenizer import EL2IDV1


class ElemBertRegressor(ElemBertEmbedding):
    def __init__(
            self,
            max_len: int = Config.MAX_LEN,
            vocab_len: int = len(EL2IDV1['el2id']),
            embed_dim: int = Config.EMBED_DIM,
            num_layers: int = Config.NUM_LAYERS,
            loss: str = "mean_squared_error",
            metrics: list[str] = ['mean_squared_error'],
            optimizer: str | tf.keras.optimizers.Optimizer = "adam",
            batch_size: int = 2,
            epochs: int = 1,
            verbose: bool = True,
        ):

        super().__init__(
            max_len=max_len,
            vocab_len=vocab_len,
            embed_dim=embed_dim,
            num_layers=num_layers,
            loss=loss,
            optimizer=optimizer,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose
        )
        self.metrics = metrics

    def make_model(self):
        inputC = layers.Input((self.max_len,), dtype=tf.int32, name="types")
        z = create_elembert_model(inputC, Config)
        e = layers.Lambda(lambda x: x[:, 0], name="clsTokenEmb")(z.output)
        f = layers.Dense(1, activation="linear",name='out_tox')(e)
        self.model = Model(inputs=z.input, outputs=f)
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        return self.model
    
    def predict(self, X):
        return self._predict(X)