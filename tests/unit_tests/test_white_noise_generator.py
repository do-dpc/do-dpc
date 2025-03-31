"""
Tests for the WhiteNoiseGenerator class.
"""

import pytest
import numpy as np

from do_dpc.control_utils.noise_generators import WhiteNoiseGenerator


def test_white_noise_initialization():
    """Test if WhiteNoiseGenerator initializes correctly with default values."""
    generator = WhiteNoiseGenerator()

    assert generator.mean.shape == (1,)
    assert generator.std.shape == (1,)
    assert generator.mean[0] == 0
    assert generator.std[0] == 1


def test_white_noise_custom_mean_std():
    """Test if WhiteNoiseGenerator correctly applies custom mean and std."""
    mean = np.array([2.0, -1.0])
    std = np.array([0.5, 1.5])
    generator = WhiteNoiseGenerator(mean=mean, std=std)

    assert generator.mean.shape == mean.shape
    assert generator.std.shape == std.shape
    assert np.array_equal(generator.mean, mean)
    assert np.array_equal(generator.std, std)


def test_white_noise_shape_mismatch():
    """Test that a ValueError is raised when mean and std have different shapes."""
    mean = np.array([1.0, 2.0])
    std = np.array([0.5])  # Mismatched shape

    with pytest.raises(ValueError, match="Mean and std must have the same shape"):
        WhiteNoiseGenerator(mean=mean, std=std)


def test_white_noise_generation():
    """Test if WhiteNoiseGenerator produces noise with correct shape and distribution."""
    mean = np.array([0.0, 0.0])
    std = np.array([1.0, 2.0])
    generator = WhiteNoiseGenerator(mean=mean, std=std)

    noise = generator.generate()

    assert noise.shape == mean.shape
    assert np.isfinite(noise).all()  # Ensure no NaN or inf values
    assert (noise >= mean - 4 * std).all() and (noise <= mean + 4 * std).all()  # Rough Gaussian bounds


def test_white_noise_reproducibility():
    """Test if WhiteNoiseGenerator produces the same noise when using the same seed."""
    seed = 42
    mean = np.array([0.0, 0.0])
    std = np.array([1.0, 2.0])

    generator1 = WhiteNoiseGenerator(mean=mean, std=std, seed=seed)
    generator2 = WhiteNoiseGenerator(mean=mean, std=std, seed=seed)

    noise1 = generator1.generate()
    noise2 = generator2.generate()

    assert np.array_equal(noise1, noise2)  # Both should generate the same output


def test_white_noise_scalar_inputs():
    """Test if scalar mean and std inputs are correctly converted to arrays."""
    generator = WhiteNoiseGenerator(mean=2.0, std=0.5)

    assert generator.mean.shape == (1,)
    assert generator.std.shape == (1,)
    assert generator.mean[0] == 2.0
    assert generator.std[0] == 0.5
