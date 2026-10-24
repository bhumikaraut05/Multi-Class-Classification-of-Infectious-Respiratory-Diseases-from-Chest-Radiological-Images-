"""Training / evaluation code.  Run from backend/:  python -m ml.<script> --help"""
import os

# The original project (and its ViT) targets legacy Keras 2 (tf-keras).
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
