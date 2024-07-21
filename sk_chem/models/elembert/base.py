from sklearn.base import BaseEstimator, TransformerMixin
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

from sk_chem.models.elembert.tokenizer import EL2IDV1, ElementTokenizer


class Config:
    MAX_LEN = 256
    BATCH_SIZE = 32
    LR = 0.001
    VOCAB_SIZE = 128
    EMBED_DIM = 32
    NUM_HEAD = 2  # used in bert model
    FF_DIM = 32  # used in bert model
    NUM_LAYERS = 2
    MNAME = "elembert_"
    MVER = "V1"


def bert_module(query, key, value, i, config):
    # Multi headed self-attention
    attention_output = layers.MultiHeadAttention(
        num_heads=config.NUM_HEAD,
        key_dim=config.EMBED_DIM // config.NUM_HEAD,
        name="encoder_{}_multiheadattention".format(i),
    )(query, key, value)
    attention_output = layers.Dropout(0.1, name="encoder_{}_att_dropout".format(i))(
        attention_output
    )
    attention_output = layers.LayerNormalization(
        epsilon=1e-6, name="encoder_{}_att_layernormalization".format(i)
    )(query + attention_output)

    # Feed-forward layer
    ffn = tf.keras.Sequential(
        [
            layers.Dense(config.FF_DIM, activation="relu"),
            layers.Dense(config.EMBED_DIM),
        ],
        name="encoder_{}_ffn".format(i),
    )
    ffn_output = ffn(attention_output)
    ffn_output = layers.Dropout(0.1, name="encoder_{}_ffn_dropout".format(i))(
        ffn_output
    )
    sequence_output = layers.LayerNormalization(
        epsilon=1e-6, name="encoder_{}_ffn_layernormalization".format(i)
    )(attention_output + ffn_output)
    return sequence_output


def get_pos_encoding_matrix(max_len: int, d_emb: int):
    pos_enc = np.array(
        [
            [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
            if pos != 0
            else np.zeros(d_emb)
            for pos in range(max_len)
        ]
    )
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
    return pos_enc


def create_elembert_model(inputs, config):
    word_embeddings = layers.Embedding(
        config.VOCAB_SIZE, config.EMBED_DIM, mask_zero=True, name="element_embdgs"
    )(inputs)
    position_embeddings = layers.Embedding(
        input_dim=config.MAX_LEN,
        output_dim=config.EMBED_DIM,
        weights=[get_pos_encoding_matrix(config.MAX_LEN, config.EMBED_DIM)],
        name="position_embedding",
    )(tf.range(start=0, limit=config.MAX_LEN, delta=1))
    embeddings = word_embeddings + position_embeddings
    encoder_output = embeddings
    for i in range(config.NUM_LAYERS):
        encoder_output = bert_module(
            encoder_output, encoder_output, encoder_output, i, config
        )
    mlm_model = Model(inputs, encoder_output, name="masked_bert_model")

    return mlm_model


class ElemBertEmbedding(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        max_len: int = Config.MAX_LEN,
        vocab_len: int = len(EL2IDV1["el2id"]),
        embed_dim: int = Config.EMBED_DIM,
        num_layers: int = Config.NUM_LAYERS,
        loss: str = "categorical_crossentropy",
        optimizer: str | tf.keras.optimizers.Optimizer = "adam",
        batch_size: int = 2,
        epochs: int = 1,
        verbose: bool = True,
    ):
        self.max_len = max_len
        self.vocab_len = vocab_len
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.loss = loss
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose

        self.word_embeddings = layers.Embedding(
            self.vocab_len, self.embed_dim, mask_zero=True, name="element_embedding"
        )

        self.position_embeddings = layers.Embedding(
            input_dim=self.max_len,
            output_dim=self.embed_dim,
            weights=[get_pos_encoding_matrix(self.max_len, self.embed_dim)],
            name="position_embedding",
        )

        self.tokenizer = ElementTokenizer()

    def make_dataset(self, X: list[list[int]], y: list | np.ndarray):
        x = tf.keras.preprocessing.sequence.pad_sequences(
            X, dtype="int32", padding="post", truncating="post", maxlen=self.max_len
        )
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.batch(self.batch_size)
        return dataset

    def make_model(self):
        raise NotImplementedError

    def fit(self, X, y):
        dataset = self.make_dataset(X, y)
        self.model = self.make_model()
        self.model.fit(dataset, epochs=self.epochs)
        return self

    def _predict(self, X):
        x = self.make_dataset(X, np.zeros(len(X)))
        return self.model.predict(x)
