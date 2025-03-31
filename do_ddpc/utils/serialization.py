"""
serialization.py

This module provides utility functions for saving and loading `dataclass` objects using NumPy `.npz` format.
It supports NumPy arrays, floats, and integers.

Functions:
    - save_dataclass_npz: Saves a dataclass object as a compressed `.npz` file.
    - load_dataclass_npz: Loads a dataclass object from a `.npz` file.
"""

import dataclasses
from typing import Type, TypeVar

import numpy as np

T = TypeVar("T")


def save_dataclass_npz(obj: T, filename: str):
    """
    Saves a dataclass object as a `.npz` file, supporting NumPy arrays, floats, and integers.

    Args:
        obj (T): The dataclass object to save.
        filename (str): The file path where the object should be stored.

    Raises:
        ValueError: If any field in the dataclass is not a `numpy.ndarray`, `float`, or `int`.

    Note:
        - This function only supports numerical attributes (`np.ndarray`, `float`, `int`).
        - Non-supported types (e.g., strings, lists, dictionaries) will raise an error.
        - Data is stored in a compressed `.npz` format to optimize storage.
    """
    data = {}
    for field in dataclasses.fields(obj):  # type: ignore
        value = getattr(obj, field.name)
        if isinstance(value, (np.ndarray, float, int)):
            data[field.name] = value
        else:
            raise ValueError(
                f"Unsupported data type {type(value)} for field '{field.name}'. "
                "Only NumPy arrays, floats, and ints are allowed."
            )

    np.savez_compressed(filename, **data)


def load_dataclass_npz(cls: Type[T], filename: str) -> T:
    """
    Loads a dataclass object from a `.npz` file, reconstructing NumPy arrays, floats, and integers.

    Args:
        cls (Type[T]): The class type of the dataclass to reconstruct.
        filename (str): The `.npz` file path from which the object should be loaded.

    Returns:
        T: The reconstructed dataclass object.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the loaded data contains unsupported types or is incompatible with the dataclass.

    Note:
        - Converts 0D NumPy arrays (e.g., `np.array(0.1)`) back to Python scalars (`float` or `int`).
        - Only fields that exist in the dataclass definition are reconstructed.
        - If a field is missing in the `.npz` file, it will raise a `TypeError` (from `dataclass` instantiation).
    """
    try:
        data = np.load(filename, allow_pickle=True)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File '{filename}' not found.") from e

    field_data = {}
    for field in dataclasses.fields(cls):  # type: ignore
        if field.name in data:
            value = data[field.name]
            if isinstance(value, np.ndarray) and value.shape == ():  # Convert 0D NumPy arrays to scalars
                value = value.item()
            field_data[field.name] = value

    return cls(**field_data)
