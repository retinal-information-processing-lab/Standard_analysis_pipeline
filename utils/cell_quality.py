"""Per-experiment cell quality, as a plain dictionary shared by every notebook.

Notebook 4a builds ``cell_quality`` — one entry per cell, like the STA results — and saves
it once per experiment in ``<output_directory>/<exp>_cell_quality.pkl``::

    cell_quality[cell_id] = {
        "rpv": 0.31,        # refractory-period-violation rate (%)
        "rpv_ok": True,     # rpv below the threshold
        "sta_ok": True,     # kept in the STA review (well-defined receptive field)
        "chirp_ok": None,   # kept in the chirp review; None = that step was not run
    }

Use it as a dict. From any later notebook::

    cell_quality = utils.load_cell_quality(params)
    good = utils.good_cells(cell_quality, cells)                 # every "_ok" that was evaluated
    good = utils.good_cells(cell_quality, cells, ["rpv_ok", "sta_ok"])   # chosen criteria
    good = [c for c in cells if cell_quality[c]["rpv_ok"]]       # or filter by hand

A criterion is ``None`` for a cell that was never reviewed for it (the step was skipped, or
the cell was not a candidate); ``None`` never passes.
"""

import os

import utils

QUALITY_CRITERIA = ("rpv_ok", "sta_ok", "chirp_ok")


def _cell_quality_path(params):
    return os.path.join(params.output_directory, f"{params.exp}_cell_quality.pkl")


def new_cell_quality(cells):
    """An empty cell_quality dict: every criterion None (not evaluated) for every cell."""
    return {
        cell: {"rpv": None, "rpv_ok": None, "sta_ok": None, "chirp_ok": None}
        for cell in cells
    }


def load_cell_quality(params):
    """The experiment's saved cell_quality dict (raises if notebook 4a was not run yet)."""
    path = _cell_quality_path(params)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No cell quality file for this experiment:\n  {path}\n"
            "Run notebook 4a (cell quality) first."
        )
    return utils.load_obj(path)


def save_cell_quality(cell_quality, params, verbose=True):
    path = _cell_quality_path(params)
    utils.save_obj(cell_quality, path)
    if verbose:
        print(f"Cell quality saved to {path}")


def good_cells(cell_quality, cells=None, criteria=None):
    """Cells whose every criterion in ``criteria`` is True, in ``cells`` order.

    ``criteria`` defaults to every criterion that was evaluated (True/False for at least one
    cell); a criterion that is None for all cells was not run and is ignored. A cell that is
    None for a used criterion fails it. ``cells`` defaults to every cell in cell_quality.
    """
    if cells is None:
        cells = list(cell_quality)
    if criteria is None:
        criteria = [
            c
            for c in QUALITY_CRITERIA
            if any(q[c] is not None for q in cell_quality.values())
        ]
    return [
        cell for cell in cells if all(cell_quality[cell][c] is True for c in criteria)
    ]


def describe_cell_quality(cell_quality):
    """Print, per criterion, how many cells pass, and return the cells passing all of them."""
    n = len(cell_quality)
    for c in QUALITY_CRITERIA:
        values = [q[c] for q in cell_quality.values()]
        if all(v is None for v in values):
            print(f"  {c:9s}: not evaluated")
        else:
            n_eval = sum(v is not None for v in values)
            print(
                f"  {c:9s}: {sum(v is True for v in values)}/{n_eval} good ({n_eval}/{n} evaluated)"
            )
    good = good_cells(cell_quality)
    print(f"  => {len(good)}/{n} cells pass every evaluated criterion:\n  {good}")
    return good
