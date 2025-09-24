"""A collection of audio filters and utilities."""

# Directly import the classes so they are easy to access.
from .iir_filter import IIRFilter
from .equal_loudness_filter import EqualLoudnessFilter

# Import the specific filter-creation functions from the butterworth_filter module.
# This allows users to call them directly from the package.
from .butterworth_filter import (
    make_lowpass,
    make_highpass,
    make_bandpass,
    make_allpass,
    make_peak,
    make_lowshelf,
    make_highshelf,
)

# The __all__ list defines the public API of the package.
# It specifies which names are imported when a user writes `from audio_filters import *`
__all__ = [
    "IIRFilter",
    "EqualLoudnessFilter",
    "make_lowpass",
    "make_highpass",
    "make_bandpass",
    "make_allpass",
    "make_peak",
    "make_lowshelf",
    "make_highshelf",
]
