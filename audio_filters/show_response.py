import argparse
from typing import Protocol, runtime_checkable
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

# --- Import local filters for demonstration ---
from butterworth_filter import make_peak
from equal_loudness_filter import EqualLoudnessFilter

# --- Type Hinting for Filters ---
@runtime_checkable
class FilterType(Protocol):
    """A protocol defining the interface for filter objects."""
    def process_block(self, block: NDArray[np.float64]) -> NDArray[np.float64]:
        ...

# --- Plotting Functions ---
def plot_response(
    samplerate: int,
    impulse_response: NDArray[np.float64],
    title: str = "Filter Response",
) -> None:
    """
    Calculates and plots the frequency and phase response of a filter.

    Args:
        samplerate: The audio sample rate in Hz.
        impulse_response: The filter's response to a single-sample impulse.
        title: The title for the plot.
    """
    # Calculate the frequency response using FFT
    fft_out = np.fft.rfft(impulse_response)
    freqs = np.fft.rfftfreq(len(impulse_response), 1.0 / samplerate)

    # Convert magnitude to decibels (dB)
    magnitude_db = 20 * np.log10(np.abs(fft_out))
    
    # Extract phase response
    phase = np.angle(fft_out)

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(title, fontsize=16)

    # Frequency Response Plot
    ax1.plot(freqs, magnitude_db, color='b')
    ax1.set_ylabel("Magnitude (dB)")
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.set_xscale('log')
    ax1.set_xlim(20, samplerate / 2)
    ax1.minorticks_on()

    # Phase Response Plot
    ax2.plot(freqs, np.unwrap(phase), color='r')
    ax2.set_ylabel("Phase (radians)")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax2.minorticks_on()
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def get_impulse_response(
    filter_obj: FilterType, length: int
) -> NDArray[np.float64]:
    """
    Generates an impulse and gets the filter's response to it.
    
    Args:
        filter_obj: The filter instance to test.
        length: The desired length of the impulse response array.

    Returns:
        The impulse response as a NumPy array.
    """
    impulse = np.zeros(length, dtype=np.float64)
    impulse[0] = 1.0  # The impulse is a single sample at the beginning
    return filter_obj.process_block(impulse)

# --- Main Demonstration ---
def main() -> None:
    """
    Runs a demonstration, creating and plotting responses for several filters.
    """
    samplerate = 44100
    impulse_length = 2048  # A suitable length for detailed FFT analysis

    print("Demonstrating a 6dB peak filter at 1000 Hz...")
    peak_filter = make_peak(1000, samplerate, gain_db=6, q_factor=1.414)
    peak_ir = get_impulse_response(peak_filter, impulse_length)
    plot_response(samplerate, peak_ir, "Peak Filter @ 1000 Hz (+6 dB)")

    print("\nDemonstrating the ISO 226:2003 Equal Loudness Filter...")
    eq_filter = EqualLoudnessFilter(samplerate)
    eq_ir = get_impulse_response(eq_filter, impulse_length)
    plot_response(samplerate, eq_ir, "Equal Loudness Filter (ISO 226:2003)")

if __name__ == "__main__":
    main()

