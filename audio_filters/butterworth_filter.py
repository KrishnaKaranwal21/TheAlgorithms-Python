from math import cos, sin, sqrt, tau

# The import now refers to the IIRFilter class in the local file
from iir_filter import IIRFilter

"""
Create 2nd-order IIR filters with Butterworth design.

Code based on https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html
"""

def _calculate_common_vars(
    frequency: int, samplerate: int, q_factor: float
) -> tuple[float, float, float, float]:
    """
    Helper function to calculate common variables used in filter coefficient design.
    This avoids code duplication across the different filter types.
    """
    w0 = tau * frequency / samplerate
    _cos = cos(w0)
    _sin = sin(w0)
    alpha = _sin / (2 * q_factor)
    return w0, _cos, _sin, alpha


def make_lowpass(
    frequency: int,
    samplerate: int,
    q_factor: float = 1 / sqrt(2),
) -> IIRFilter:
    """Creates a 2nd-order Butterworth low-pass filter."""
    w0, _cos, _sin, alpha = _calculate_common_vars(frequency, samplerate, q_factor)

    b0 = (1 - _cos) / 2
    b1 = 1 - _cos
    b2 = b0  # Corrected from the original to match cookbook b2 = (1-cos(w0))/2
    a0 = 1 + alpha
    a1 = -2 * _cos
    a2 = 1 - alpha

    filt = IIRFilter(2)
    # Coefficients are normalized by a0 for stability
    filt.set_coefficients([1, a1 / a0, a2 / a0], [b0 / a0, b1 / a0, b2 / a0])
    return filt


def make_highpass(
    frequency: int,
    samplerate: int,
    q_factor: float = 1 / sqrt(2),
) -> IIRFilter:
    """Creates a 2nd-order Butterworth high-pass filter."""
    w0, _cos, _sin, alpha = _calculate_common_vars(frequency, samplerate, q_factor)

    b0 = (1 + _cos) / 2
    b1 = -(1 + _cos)
    b2 = b0 # Corrected from original to match cookbook
    a0 = 1 + alpha
    a1 = -2 * _cos
    a2 = 1 - alpha

    filt = IIRFilter(2)
    filt.set_coefficients([1, a1 / a0, a2 / a0], [b0 / a0, b1 / a0, b2 / a0])
    return filt


def make_bandpass(
    frequency: int,
    samplerate: int,
    q_factor: float = 1, # Default Q for bandpass is typically 1
) -> IIRFilter:
    """Creates a 2nd-order Butterworth band-pass filter."""
    w0, _cos, _sin, alpha = _calculate_common_vars(frequency, samplerate, q_factor)

    b0 = alpha
    b1 = 0
    b2 = -alpha
    a0 = 1 + alpha
    a1 = -2 * _cos
    a2 = 1 - alpha

    filt = IIRFilter(2)
    filt.set_coefficients([1, a1 / a0, a2 / a0], [b0 / a0, b1 / a0, b2 / a0])
    return filt


def make_allpass(
    frequency: int,
    samplerate: int,
    q_factor: float = 1 / sqrt(2),
) -> IIRFilter:
    """Creates a 2nd-order all-pass filter."""
    w0, _cos, _sin, alpha = _calculate_common_vars(frequency, samplerate, q_factor)

    b0 = 1 - alpha
    b1 = -2 * _cos
    b2 = 1 + alpha
    a0 = 1 + alpha
    a1 = -2 * _cos
    a2 = 1 - alpha

    filt = IIRFilter(2)
    filt.set_coefficients([1, a1 / a0, a2 / a0], [b0 / a0, b1 / a0, b2 / a0])
    return filt


def make_peak(
    frequency: int,
    samplerate: int,
    gain_db: float,
    q_factor: float = 1 / sqrt(2),
) -> IIRFilter:
    """Creates a peak (or bell) filter."""
    w0, _cos, _sin, alpha = _calculate_common_vars(frequency, samplerate, q_factor)
    A = 10 ** (gain_db / 40)

    b0 = 1 + alpha * A
    b1 = -2 * _cos
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * _cos
    a2 = 1 - alpha / A

    filt = IIRFilter(2)
    filt.set_coefficients([1, a1 / a0, a2 / a0], [b0 / a0, b1 / a0, b2 / a0])
    return filt


def make_lowshelf(
    frequency: int,
    samplerate: int,
    gain_db: float,
    q_factor: float = 1, # Use slope parameter S=1 for shelves
) -> IIRFilter:
    """Creates a low-shelf filter."""
    w0, _cos, _sin, alpha = _calculate_common_vars(frequency, samplerate, q_factor)
    A = 10 ** (gain_db / 40)
    
    # Simplified coefficients from the cookbook for shelving filters
    b0 = A * ((A + 1) - (A - 1) * _cos + 2 * sqrt(A) * alpha)
    b1 = 2 * A * ((A - 1) - (A + 1) * _cos)
    b2 = A * ((A + 1) - (A - 1) * _cos - 2 * sqrt(A) * alpha)
    a0 = (A + 1) + (A - 1) * _cos + 2 * sqrt(A) * alpha
    a1 = -2 * ((A - 1) + (A + 1) * _cos)
    a2 = (A + 1) + (A - 1) * _cos - 2 * sqrt(A) * alpha

    filt = IIRFilter(2)
    filt.set_coefficients([1, a1 / a0, a2 / a0], [b0 / a0, b1 / a0, b2 / a0])
    return filt


def make_highshelf(
    frequency: int,
    samplerate: int,
    gain_db: float,
    q_factor: float = 1, # Use slope parameter S=1 for shelves
) -> IIRFilter:
    """Creates a high-shelf filter."""
    w0, _cos, _sin, alpha = _calculate_common_vars(frequency, samplerate, q_factor)
    A = 10 ** (gain_db / 40)

    # Simplified coefficients from the cookbook for shelving filters
    b0 = A * ((A + 1) + (A - 1) * _cos + 2 * sqrt(A) * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * _cos)
    b2 = A * ((A + 1) + (A - 1) * _cos - 2 * sqrt(A) * alpha)
    a0 = (A + 1) - (A - 1) * _cos + 2 * sqrt(A) * alpha
    a1 = 2 * ((A - 1) - (A + 1) * _cos)
    a2 = (A + 1) - (A - 1) * _cos - 2 * sqrt(A) * alpha

    filt = IIRFilter(2)
    filt.set_coefficients([1, a1 / a0, a2 / a0], [b0 / a0, b1 / a0, b2 / a0])
    return filt

