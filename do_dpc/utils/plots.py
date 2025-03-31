"""
Module for generating plots.
"""

import base64
import io
import os

import matplotlib.pyplot as plt
import numpy as np
from IPython import display as ipythondisplay
from IPython.display import HTML

from do_dpc.control_utils.control_structs import InputOutputTrajectory
from do_dpc.utils.logging_config import path_manager


def plot_training_data(training_data: InputOutputTrajectory):
    """
    Plots the input (u) and output (y) signals from training_data.

    Args:
        training_data: An object with attributes `u` (m x n_samples) and `y` (p x n_samples),
                       both being NumPy arrays.
    """
    n_samples = training_data.u.shape[1]  # Number of time steps
    time = range(n_samples)  # X-axis (sample indices)

    _, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Plot input u
    for i in range(training_data.u.shape[0]):
        axes[0].plot(time, training_data.u[i], label=f"Input {i + 1}")
    axes[0].set_title("Input (u)")
    axes[0].legend()
    axes[0].grid()

    # Plot output y
    for i in range(training_data.y.shape[0]):
        axes[1].plot(time, training_data.y[i], label=f"Output {i + 1}")
    axes[1].set_title("Output (y)")
    axes[1].legend()
    axes[1].grid()

    axes[1].set_xlabel("Time Steps")
    plt.tight_layout()
    plt.show()


def generate_rocket_actuator_plot(actuator_data: np.ndarray, plot_title: str):
    """
    Generate a plot of the actuators.

    Args:
        actuator_data: Array of the actuators with length.
        plot_title: Title of the plot
    """
    plt.figure()
    plt.title(plot_title)
    plt.grid()
    plt.plot(actuator_data[:, 0], label="Fe")
    plt.plot(actuator_data[:, 1], label="Fs")
    plt.plot(actuator_data[:, 2], label="phi")
    plt.legend()
    plt.show()


def generate_rocket_state_plot(state_data: np.ndarray, plot_title: str):
    """
    Generate a plot of the states.

    Args:
        state_data: Array of the states with length 6
        plot_title: Title of the plot
    """
    plt.figure()
    plt.title(plot_title)
    plt.grid()
    plt.plot(state_data[:, 0], label="x")
    plt.plot(state_data[:, 1], label="y")
    plt.plot(state_data[:, 4], label="theta")
    plt.plot(state_data[:, 3], label="y_dot")
    plt.legend()
    plt.show()


def show_video(video_title: str):
    """
    Displays a video in the Jupyter Notebook by embedding it as a base64-encoded data URL.

    Args:
        video_title (str): The title of the video (without extension), used to construct the filename.

    Raises:
        FileNotFoundError: If the specified video file does not exist in the video folder.
        Exception: If there's an error during file reading or base64 encoding.
    """
    video_folder = path_manager.get_video_path()

    video_filename = f"{video_title}-episode-0.mp4"
    video_path = os.path.join(video_folder, video_filename)

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"The video file '{video_filename}' does not exist in the folder '{video_folder}'.")

    try:
        # pylint: disable=consider-using-with
        video = io.open(video_path, "r+b").read()
        encoded = base64.b64encode(video)
        # pylint: disable=consider-using-f-string
        ipythondisplay.display(
            HTML(
                data="""<video alt="test" autoplay
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>""".format(
                    encoded.decode("ascii")
                )
            )
        )

    # pylint: disable=broad-exception-raised
    except Exception as e:
        raise Exception(f"An error occurred while processing the video '{video_filename}': {str(e)}") from e
