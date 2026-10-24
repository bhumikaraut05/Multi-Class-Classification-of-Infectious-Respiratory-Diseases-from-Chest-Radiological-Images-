import os

# The original project targets legacy Keras 2 (tf-keras); keep that for every entry point.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
