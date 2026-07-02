"""
Shared utilities for the standard analysis pipeline.

This package was split from a single ``utils.py`` into themed submodules so the
code is easier to navigate (e.g. from an online notebook viewer). Every public
name is re-exported here, so existing code keeps working unchanged::

    import utils
    utils.load_obj(...)          # still works
    from utils import load_obj   # still works

Submodules (import them directly if you want to see where a function lives):
    general        generic raster/PSTH plotting helper
    io             recording/trigger/spike loaders, directory & prompt helpers
    preprocessing  raw-data preprocessing (symlinks, filtering, spike sorting, pickle I/O)
    checkerboard   checkerboard stimulus reconstruction & raster building
    sta            spike-triggered-average / receptive-field analysis
    clustering     cell-typing clustering helpers
    id_card        per-cell ID-card helpers
    vec            vec-file "sequence" analysis (rasters/PSTH per repetition)
    holography     holographic registration & legacy STA analyses
"""

# Order follows the dependency graph (deps first): io depends on preprocessing,
# holography depends on sta. Each submodule also imports its own dependencies,
# so this order only avoids redundant work.
from .general import *
from .preprocessing import *
from .io import *
from .checkerboard import *
from .sta import *
from .holography import *
from .clustering import *
from .id_card import *
from .vec import *