"""Pyomo reader module."""
import os
import importlib
import importlib.util
from galini.pyomo.osil_reader import read_osil
from galini.error import (
    InvalidFileExtensionError,
    InvalidPythonInputError,
)


def read_python(filename, **_kwargs):
    """Read Pyomo model from Python file.

    Arguments
    ---------
    filename : str
        the input file.

    Returns
    -------
    ConcreteModel
        Pyomo concrete model.
    """
    spec = importlib.util.spec_from_file_location('_input_model_module', filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, 'get_pyomo_model'):
        raise InvalidPythonInputError('invalid python input')
    return module.get_pyomo_model()


READER_BY_EXT = {
    '.osil': read_osil,
    '.xml': read_osil,
    '.py': read_python,
}



def read_pyomo_model(filename, **kwargs):
    """Read Pyomo model from file.

    Arguments
    ---------
    filename : str
        the input file.

    Returns
    -------
    ConcreteModel
        Pyomo concrete model.
    """
    _, ext = os.path.splitext(filename)
    if ext not in READER_BY_EXT:
        raise InvalidFileExtensionError('invalid extension')
    reader = READER_BY_EXT[ext]
    return reader(filename, **kwargs)
