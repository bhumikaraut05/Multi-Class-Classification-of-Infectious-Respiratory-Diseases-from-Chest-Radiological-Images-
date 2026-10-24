"""Candidate architectures.  The same ``build_model`` is used for training and by the API.

Notes on provenance (important for the write-up):
* ``vit``           - the architecture from the original project (app/vit_arch.py, 89.7 M parameters).
* ``vgg19`` / ``resnet50`` - standard Keras Applications backbones + a small classification head.
* ``baseline_cnn`` / ``custom_cnn_sa`` - re-implementations written for this pipeline, because the original
  notebooks were NOT in the supplied ZIP.  They follow the idea (plain CNN / CNN with spatial attention) but are
  not guaranteed to match the original layer-by-layer.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Input, Model, layers

from app.config import CLASS_NAMES

NUM_CLASSES = len(CLASS_NAMES)

# Explanation method per architecture.  For CNNs the named layer is used for Grad-CAM.
EXPLAIN = {
    "baseline_cnn": ("grad_cam", "last_conv"),
    "custom_cnn_sa": ("grad_cam", "last_conv"),
    "vgg19": ("grad_cam", "block5_conv4"),
    "resnet50": ("grad_cam", "conv5_block3_out"),
    "vit": ("attention_rollout", None),
}


def _conv_block(x, filters, name, bn=True):
    for k in (1, 2):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=not bn, name=f"{name}_conv{k}")(x)
        if bn:
            x = layers.BatchNormalization(name=f"{name}_bn{k}")(x)
        x = layers.Activation("relu", name=f"{name}_relu{k}")(x)
    return x


def _spatial_attention(x, name):
    """CBAM-style spatial attention: channel-wise avg+max -> 7x7 conv -> sigmoid mask applied to the features."""
    avg = layers.Lambda(lambda t: tf.reduce_mean(t, axis=-1, keepdims=True), name=f"{name}_avg")(x)
    mx = layers.Lambda(lambda t: tf.reduce_max(t, axis=-1, keepdims=True), name=f"{name}_max")(x)
    mask = layers.Conv2D(1, 7, padding="same", activation="sigmoid", name=f"{name}_map")(
        layers.Concatenate(name=f"{name}_cat")([avg, mx]))
    return layers.Multiply(name=f"{name}_apply")([x, mask])


def _head(x, dropout=0.4):
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation="relu", name="fc")(x)
    x = layers.Dropout(dropout, name="drop")(x)
    return layers.Dense(NUM_CLASSES, activation="softmax", name="probs")(x)


def build_model(name: str, input_size: int = 224, pretrained: bool = False) -> Model:
    inp = Input((input_size, input_size, 3), name="image")
    if name == "baseline_cnn":
        x = inp
        for i, f in enumerate((32, 64, 128), 1):
            x = layers.Conv2D(f, 3, padding="same", activation="relu", name=f"b{i}_conv")(x)
            x = layers.MaxPooling2D(name=f"b{i}_pool")(x)
        x = layers.Conv2D(128, 3, padding="same", activation="relu", name="last_conv")(x)
        return Model(inp, _head(x), name=name)
    if name == "custom_cnn_sa":
        x = inp
        for i, f in enumerate((32, 64, 128), 1):
            x = _conv_block(x, f, f"b{i}")
            x = layers.MaxPooling2D(name=f"b{i}_pool")(x)
        x = _spatial_attention(x, "sa3")
        x = _conv_block(x, 256, "b4")
        x = _spatial_attention(x, "sa4")
        x = layers.Activation("linear", name="last_conv")(x)
        return Model(inp, _head(x), name=name)
    if name in ("vgg19", "resnet50"):
        w = "imagenet" if pretrained else None
        app = tf.keras.applications
        base = (app.VGG19 if name == "vgg19" else app.ResNet50)(include_top=False, weights=w, input_tensor=inp)
        return Model(inp, _head(base.output), name=name)
    if name == "vit":
        from app.vit_arch import build_vit
        clf, _att = build_vit()
        return clf
    raise ValueError(f"Unknown model '{name}'. Choose from {list(EXPLAIN)}")
