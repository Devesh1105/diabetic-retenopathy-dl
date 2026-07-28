"""Shared pytest setup."""

import os

# Stop albumentations from phoning home for a version check during tests.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
# Keep TensorFlow's C++ logging out of the test output.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
