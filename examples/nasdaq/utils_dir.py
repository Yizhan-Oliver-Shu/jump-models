"""
Helpers for working with file directories.

Useful for all scripts/notebooks in this folder. 
Please ensure the file structure under `example/Nasdaq` is preserved 
in its original form for everything to function properly.
"""

import sys, os

def get_curr_dir():
    """
    Return the current directory of this `get_data.py` file.
    """
    return os.path.dirname(os.path.abspath(__file__))

def include_home_dir():
    """
    Add the project's home directory to `sys.path`.

    This function ensures that the home directory of the project is included in 
    `sys.path` to allow imports from other parts of the project. For this to work 
    correctly, the script must be placed in the `example/Nasdaq/` folder.
    """
    curr_dir = get_curr_dir()
    home_dir = os.path.dirname(os.path.dirname(curr_dir))
    sys.path.append(home_dir)
    return