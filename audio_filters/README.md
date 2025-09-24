Python Audio Filters
A collection of classic digital audio filters implemented in Python, designed for clarity, performance, and accuracy. This library provides building blocks for audio processing tasks, including common Butterworth filters and a psychoacoustically-accurate Equal Loudness filter.

Features
IIR Filter Core: A robust and efficient N-order IIR filter class optimized with collections.deque.

Butterworth Filters: A full set of 2nd-order Butterworth filter designs based on the popular "Audio EQ Cookbook," including Low-pass, High-pass, Band-pass, Peak, and Shelving filters.

Equal Loudness Filter: A specialized filter that compensates for the non-linear nature of human hearing, based on the modern ISO 226:2003 standard.

Visualization Tools: Includes a script to plot the frequency and phase response of any filter using matplotlib.

Tested: Core filter implementations are validated against scipy using the pytest framework.

Installation
To get started, clone the repository and install the project along with its dependencies. This project uses pyproject.toml for package management.

Clone the repository:

git clone <repository-url>
cd audio-filter

Install the library:
This command installs the core dependencies (numpy, yulewalker).

pip install .

Install development dependencies (optional):
To run the tests and visualization script, you need to install the optional "dev" dependencies.

pip install ".[dev]"

Usage
You can easily create and use filters from the library. The following example creates a 2nd-order low-pass filter and visualizes its frequency response.

# --- main.py ---
from butterworth_filter import make_lowpass
from show_response import show_frequency_response, show_phase_response

# Filter parameters
SAMPLERATE_HZ = 48000
CUTOFF_HZ = 1000

# 1. Create the filter
lowpass_filter = make_lowpass(CUTOFF_HZ, SAMPLERATE_HZ)

# 2. Visualize the frequency and phase response
print("Displaying frequency response for the low-pass filter...")
show_frequency_response(lowpass_filter, SAMPLERATE_HZ)

print("Displaying phase response for the low-pass filter...")
show_phase_response(lowpass_filter, SAMPLERATE_HZ)

Running the Demonstration
A built-in demonstration script is included to visualize several of the library's filters.

python show_response.py

This will display the response plots for a sample peak filter and the Equal Loudness filter.

Testing
The project includes a test suite to ensure the filter implementations are correct. The tests compare the generated filter coefficients against scipy's trusted implementations.

To run the tests, first ensure you have installed the development dependencies, then run pytest:

pytest
