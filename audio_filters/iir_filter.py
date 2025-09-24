from __future__ import annotations
from collections import deque
import numpy as np
from numpy.typing import NDArray

class IIRFilter:
    r"""
    A robust, N-Order IIR filter based on the Direct Form 1 structure.

    This class is optimized for performance using `collections.deque` for its
    internal state and supports both single-sample and block-based processing.
    Coefficients are automatically normalized for stability.
    """

    def __init__(self, order: int) -> None:
        """
        Initializes the IIR filter.

        Args:
            order: The order of the filter (must be 1 or greater).
        """
        if order < 1:
            raise ValueError("Filter order must be at least 1.")
        self.order = order
        
        # Initialize with default pass-through coefficients
        self.a_coeffs: list[float] = [1.0] + [0.0] * order
        self.b_coeffs: list[float] = [1.0] + [0.0] * order

        # History buffers for the filter's internal state
        self.input_history = deque([0.0] * self.order, maxlen=self.order)
        self.output_history = deque([0.0] * self.order, maxlen=self.order)

    def set_coefficients(self, a_coeffs: list[float], b_coeffs: list[float]) -> None:
        """
        Sets and normalizes the IIR filter coefficients.

        The coefficients are normalized by a_coeffs[0] to ensure stability
        and simplify the processing loop.

        Args:
            a_coeffs: The denominator coefficients [a0, a1, ...].
            b_coeffs: The numerator coefficients [b0, b1, ...].
        """
        if len(a_coeffs) != self.order + 1 or len(b_coeffs) != self.order + 1:
            msg = (
                f"Expected coefficient lists of length {self.order + 1}, "
                f"but got lengths {len(a_coeffs)} and {len(b_coeffs)}"
            )
            raise ValueError(msg)
        
        a0 = a_coeffs[0]
        if a0 == 0:
            raise ValueError("The first 'a' coefficient (a0) cannot be zero.")

        # Normalize all coefficients by a0 for stability and efficiency
        self.a_coeffs = [a / a0 for a in a_coeffs]
        self.b_coeffs = [b / a0 for b in b_coeffs]

    def process(self, sample: float) -> float:
        """
        Processes a single audio sample through the filter.

        Args:
            sample: A single float representing the audio sample.

        Returns:
            The processed audio sample.
        """
        # Calculate the feedforward part (from input history)
        feedforward = sum(b * x for b, x in zip(self.b_coeffs[1:], self.input_history))
        
        # Calculate the feedback part (from output history)
        feedback = sum(a * y for a, y in zip(self.a_coeffs[1:], self.output_history))
        
        # Combine with the current sample (b0) and subtract feedback
        # Note: a0 is 1.0 due to normalization in set_coefficients
        result = self.b_coeffs[0] * sample + feedforward - feedback

        # Update the history buffers for the next sample
        self.input_history.appendleft(sample)
        self.output_history.appendleft(result)

        return result

    def process_block(self, block: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Processes a block (NumPy array) of audio samples.

        This is the preferred method for processing audio files or buffers
        as it is more efficient than calling `process()` in a Python loop.

        Args:
            block: A NumPy array of audio samples.

        Returns:
            A NumPy array containing the processed samples.
        """
        output_block = np.zeros_like(block)
        for i, sample in enumerate(block):
            output_block[i] = self.process(sample)
        return output_block

    def clear(self) -> None:
        """
        Clears the filter's internal history buffers.

        This resets the filter to its initial state, which is useful for
        processing discontinuous audio streams.
        """
        self.input_history.clear()
        self.output_history.clear()
        self.input_history.extend([0.0] * self.order)
        self.output_history.extend([0.0] * self.order)

