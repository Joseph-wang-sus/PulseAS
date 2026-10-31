"""PulseAS: PPG-based aortic stenosis analysis."""
import os
import sys
from pathlib import Path

# Conda on Windows resolves SciPy's L-BFGS-B DLLs through Library/bin.
# Keep the handle alive, as in the original study runners.
_dll_handle = None
_dll_directory = Path(sys.prefix) / 'Library/bin'
if os.name == 'nt' and _dll_directory.is_dir():
    os.environ['PATH'] = str(_dll_directory) + os.pathsep + os.environ.get('PATH','')
    _dll_handle = os.add_dll_directory(str(_dll_directory))
__version__ = "0.1.0"
