from json import loads
from pathlib import Path
from typing import Any, cast

import numpy as np
from yulewalker import yulewalk

# Corrected imports for the single-folder project structure
from butterworth_filter import make_highpass
from iir_filter import IIRFilter

# --- Helper Function for Data Loading ---
# By moving the data loading into a helper function, we ensure the JSON file
# is read only once and the logic is neatly encapsulated.
_LOUDNESS_DATA: dict[str, Any] | None = None

def _get_loudness_data() -> dict[str, Any]:
    """Loads the loudness curve data from the JSON file on first call."""
    global _LOUDNESS_DATA
    if _LOUDNESS_DATA is None:
        data_path = Path(__file__).resolve().parent / "loudness_curve.json"
        _LOUDNESS_DATA = cast(dict[str, Any], loads(data_path.read_text()))
    return _LOUDNESS_DATA


# --- Main Filter Class ---
class EqualLoudnessFilter:
    r"""
    An equal-loudness filter which compensates for the human ear's non-linear response
    to sound, making perceived loudness more uniform across frequencies.

    This filter works by cascading two separate filters:
    1. A high-order Yule-Walker filter designed to match the inverse of the
       ISO 226:2003 equal-loudness curve.
    2. A gentle high-pass filter to roll off subsonic frequencies below the
       range of human hearing.
    """

    def __init__(
        self,
        samplerate: int = 44100,
        yulewalk_order: int = 10,
        highpass_cutoff: int = 150,
    ) -> None:
        """
        Initializes the EqualLoudnessFilter.

        Args:
            samplerate: The audio sample rate in Hz.
            yulewalk_order: The order of the main Yule-Walker filter.
                            Higher orders provide a more accurate frequency response
                            but are more computationally expensive.
            highpass_cutoff: The cutoff frequency for the subsonic high-pass filter.
        """
        if samplerate <= 0:
            raise ValueError("Sample rate must be a positive number.")

        self.yulewalk_filter = IIRFilter(yulewalk_order)
        self.butterworth_filter = make_highpass(highpass_cutoff, samplerate)

        data = _get_loudness_data()

        # Pad the curve data to the Nyquist frequency to ensure the filter is
        # well-behaved at the upper end of the spectrum.
        curve_freqs = np.array(data["frequencies"] + [samplerate / 2])
        curve_gains = np.array(data["gains"] + [140.0]) # Add a high gain point

        # Normalize frequencies to the Nyquist frequency (where 1.0 = samplerate / 2)
        freqs_normalized = curve_freqs / (samplerate / 2)
        
        # Invert the loudness curve and normalize its minimum to 0dB to create
        # the target response for the filter.
        gains_normalized = np.power(10, (np.min(curve_gains) - curve_gains) / 20)

        # The `yulewalk` function returns coefficients as (b, a).
        b_coeffs, a_coeffs = yulewalk(yulewalk_order, freqs_normalized, gains_normalized)
        
        # **CRITICAL BUG FIX:** The IIRFilter expects (a, b) coefficients.
        # The original code had them reversed.
        self.yulewalk_filter.set_coefficients(a_coeffs, b_coeffs)

    def process(self, sample: float) -> float:
        """Processes a single audio sample through the cascaded filters."""
        # Process through the main loudness compensation filter first.
        compensated_sample = self.yulewalk_filter.process(sample)
        # Then, clean up any subsonic rumble with the high-pass filter.
        return self.butterworth_filter.process(compensated_sample)

