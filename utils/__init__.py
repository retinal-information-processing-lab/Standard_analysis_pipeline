# This file is a re-export facade: it intentionally pulls every submodule's
# public names via star imports, so "unused import" (F401) and "star import"
# (F403) warnings are silenced for this file only.
# ruff: noqa: F401, F403
"""
Shared utilities for the standard analysis pipeline.

This package was split from a single ``utils.py`` into themed submodules so the
code is easier to navigate (e.g. from an online notebook viewer). Every public
name is re-exported here, so existing code keeps working unchanged::

    import utils
    utils.load_obj(...)          # still works
    from utils import load_obj   # still works

Shared submodules (re-exported here):
    general        generic raster/PSTH plotting helper
    loading        recording/trigger/spike loaders, directory & prompt helpers
    preprocessing  raw-data preprocessing (symlinks, filtering, spike sorting, pickle I/O)
    checkerboard   checkerboard stimulus reconstruction & raster building
    sta            spike-triggered-average / receptive-field analysis
    cell_typing    clustering / cell-typing helpers (chirp PSTH + STA -> clusters)
    RPV_analysis   refractory-period-violation (RPV) spike-quality metrics
    vec            vec-file "sequence" analysis (rasters/PSTH per repetition)
    holography     holographic registration & legacy STA analyses

Per-notebook companion submodules (NOT re-exported; import explicitly, e.g.
``from utils import drifting_gratings as analysis``):
    analyse_checkerboard  <- 2-Analyse_Checkerboard.ipynb
    drifting_gratings     <- 3-Drifting_Gratings.ipynb
    chirp                 <- 4-Chirp+Cell Typing.ipynb (chirp rasters)
    cell_id               <- 5_Cell_ID_card.ipynb
"""

# Order follows the dependency graph (deps first): loading depends on preprocessing,
# holography depends on sta. Each submodule also imports its own dependencies,
# so this order only avoids redundant work.
from .general import *
from .preprocessing import *
from .loading import *
from .checkerboard import *
from .sta import *
from .holography import *
from .cell_typing import *
from .RPV_analysis import *
from .vec import *