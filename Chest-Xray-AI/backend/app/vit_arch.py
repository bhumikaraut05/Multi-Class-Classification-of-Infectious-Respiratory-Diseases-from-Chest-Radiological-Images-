"""Vision Transformer architecture, ported from the research notebook.

Source: "Vision Transformer with Multi-head self attention ..." notebook,
cells defining ``TokenAndPositionEmbedding`` and ``VisionTransformer`` /
``VisionTransformer_with_attention`` (final model, saved as VitB16_DownSized.h5).

The structure (including quirks such as ``Dense(1, activation='softmax')``
in the attention pooling) is preserved on purpose so the trained weights load
in the same layer order. Do not "fix" it without retraining.
"""
from __future__ import annotations

import os

# The notebook was written for Keras 2 (tf.keras). Force the legacy Keras so
# graph-building with raw tf ops and legacy .h5 weights behave identically.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import tensorflow as tf  # noqa: E402
from tensorflow.keras import Input, Model  # noqa: E402
from tensorflow.keras.layers import (  # noqa: E402
    Add, Concatenate, Dense, Dropout, GlobalAveragePooling1D,
    LayerNormalization, MultiHeadAttention, Reshape,
)

# Hyper-parameters of the final model (notebook cell 54).
IMAGE_SIZE = 224
PATCH_SIZE = 20
NUM_CLASSES = 5
NUM_BLOCKS = 5
PROJECTION_DIM = 768
NUM_HEADS = 5
MLP_UNITS = [3072, 768]
DROPOUT_RATE = 0.1
GRID = IMAGE_SIZE // PATCH_SIZE  # 11 -> 121 patches


class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    """Prepends a learned class token and adds learned position embeddings."""

    def __init__(self, projection_dim: int = PROJECTION_DIM, **kwargs):
        super().__init__(**kwargs)
        self.projection_dim = projection_dim

    def build(self, input_shape):
        self.pos_embed = self.add_weight(
            name="pos_embed", shape=(1, input_shape[1] + 1, self.projection_dim),
            initializer="zeros", trainable=True)
        self.class_token = self.add_weight(
            name="class_token", shape=(1, 1, self.projection_dim),
            initializer="zeros", trainable=True)
        super().build(input_shape)

    def call(self, patches):
        cls = tf.broadcast_to(self.class_token, [tf.shape(patches)[0], 1, self.projection_dim])
        return tf.concat([cls, patches], axis=1) + self.pos_embed

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"projection_dim": self.projection_dim})
        return cfg


def build_vit() -> tuple[Model, Model]:
    """Return ``(classifier, attention_model)`` sharing the same layers/weights.

    ``classifier`` maps image -> class probabilities.
    ``attention_model`` maps image -> [probabilities, attn_block_1..attn_block_N],
    each attn tensor shaped (batch, heads, tokens, tokens).
    """
    inputs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    patches = tf.image.extract_patches(
        images=inputs, sizes=[1, PATCH_SIZE, PATCH_SIZE, 1],
        strides=[1, PATCH_SIZE, PATCH_SIZE, 1], rates=[1, 1, 1, 1], padding="VALID")
    patches = Reshape((patches.shape[1] * patches.shape[2], PATCH_SIZE * PATCH_SIZE * 3))(patches)
    tokens = Dense(PROJECTION_DIM)(patches)
    tokens = TokenAndPositionEmbedding(PROJECTION_DIM)(tokens)

    attn_scores = []
    for _ in range(NUM_BLOCKS):
        x = LayerNormalization(epsilon=1e-6)(tokens)
        attn_out, scores = MultiHeadAttention(
            num_heads=NUM_HEADS, key_dim=PROJECTION_DIM, dropout=DROPOUT_RATE
        )(x, x, return_attention_scores=True)
        attn_scores.append(scores)
        x = Add()([attn_out, tokens])
        x = LayerNormalization(epsilon=1e-6)(x)
        mlp = Dense(MLP_UNITS[0], activation="relu")(x)
        mlp = Dropout(DROPOUT_RATE)(mlp)
        mlp = Dense(MLP_UNITS[1])(mlp)
        tokens = Add()([mlp, x])

    # Attention pooling exactly as trained (softmax over a size-1 axis == 1.0).
    pool_w = Dense(1, activation="softmax")(tokens)
    pooled = tf.reduce_sum(tokens * pool_w, axis=1)
    pooled = Dense(2096, activation="relu")(pooled)
    pooled = Dropout(0.4)(pooled)
    pooled = Dense(2096, activation="relu")(pooled)
    gap = GlobalAveragePooling1D()(tokens)
    pooled = Concatenate()([pooled, gap])
    probs = Dense(NUM_CLASSES, activation="softmax")(pooled)

    classifier = Model(inputs, probs, name="vit_classifier")
    attention_model = Model(inputs, [probs, *attn_scores], name="vit_attention")
    return classifier, attention_model
