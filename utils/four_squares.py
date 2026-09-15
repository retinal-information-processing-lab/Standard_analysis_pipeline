"""The "four squares" stimulus: geometry, frame generation, and an RF-vs-response check.

The stimulus flashes a white square in each of the four cardinal directions just OUTSIDE
the MEA (top, right, bottom, left), one square at a time. The MEA sits in the centre of the
display; each square abuts one edge of the MEA footprint, extends outward by the MEA
half-size, and spans the MEA's width along the parallel edge.

It is a spatial sanity check: a cell whose receptive field (RF) is at the top should respond
most to the top square. This module builds the stimulus (NB B) and, given the per-square
responses and the RFs from the checkerboard STA, plots the check (NB A).

Geometry assumes a SQUARE display (as in the normalised [-1, 1]^2 framing). It is validated
on MEA 2 (square DMD); on a non-square rig the square placement is only approximate.
"""

import numpy as np
import matplotlib.pyplot as plt

import params

# Canonical order of the four squares and, for each, the unit direction it lies in
# (image convention: +x right, +y up).
FOUR_SQUARES = [
    ("top", (0, 1)),
    ("right", (1, 0)),
    ("bottom", (0, -1)),
    ("left", (-1, 0)),
]

# ---------------------------------------------------------------------------
# Standard stimulus preamble — reserved sequence ids.
#
# By convention every stimulus bin starts with the same leading frames, and every recorded
# vec plays the same short preamble before the real stimulus, so each recording carries its
# own display checks:
#
#   Standard bin (frame index -> content):
#     0 : grey   (mean-luminance reference)
#     1 : black  (gap between squares)
#     2 : top square    3 : right square    4 : bottom square    5 : left square
#     6.. : the real stimulus frames
#
#   Standard vec: 1 s black (key 0, ignored), then each square = 1 s white + 1 s black under
#   the square's key (so a cell's OFF response to a square is counted for THAT square and
#   never spills into the next one), THEN the real stimulus.
#
# The preamble uses RESERVED sequence ids in the 999x block so they can never collide with a
# real stimulus's sequence ids (which start at 1 and rarely exceed ~100). Keys are still
# ``sequence_id * 10**n_digit_for_rep + repetition``, e.g. top square rep 0 -> 99910000 with
# 4 repetition digits. The preamble MUST use the same number of repetition digits as the
# stimulus it is prepended to (the analysis decodes every key with one n_digit_for_rep):
# prepend_standard_preamble takes care of it. A real stimulus keeps its own 1..N ids
# alongside, untouched.
#
# The F orientation test is deliberately NOT part of the preamble: it exists to debug the
# user's OWN stimulus-making code, so it must be produced by that code (own F frame in the
# bin, own vec rows). 9999 is the reserved id to give it so it never clashes with anything.
F_CHECK_SEQUENCE_ID = 9999
FOUR_SQUARES_SEQUENCE_IDS = {"top": 9991, "right": 9992, "bottom": 9993, "left": 9994}

# The four squares' sequence keys as the analysis sees them (the id part of the vec key),
# in canonical order — the default the response plot expects.
FOUR_SQUARES_KEYS = [str(FOUR_SQUARES_SEQUENCE_IDS[name]) for name, _ in FOUR_SQUARES]

# Frame index of each standard leading frame in the bin.
STANDARD_BIN_FRAME_INDEX = {
    "grey": 0,
    "black": 1,
    "top": 2,
    "right": 3,
    "bottom": 4,
    "left": 5,
}
N_STANDARD_BIN_FRAMES = len(
    STANDARD_BIN_FRAME_INDEX
)  # real stimulus frames start after these


def mea_extent_on_display(mea=None):
    """MEA half-extent as a fraction of the display half-width.

    The display image spans normalised [-1, 1] and is centred on the MEA. The MEA is a
    square of ``n_electrodes * mea_spacing`` µm; the DMD spans ``size_dmd * pxl_size_dmd``
    µm. The MEA therefore occupies ``mea_um / dmd_um`` of the display, which is also its
    half-extent in normalised units (half-width 1 <-> half the display).

    Returns a single fraction, using the display's x size (square-display assumption).
    """
    if mea is None:
        mea = params.MEA
    rig = params.get_rig_params(mea)
    size_dmd, pxl = rig["size_dmd"], rig["pxl_size_dmd"]
    if size_dmd is None or pxl is None:
        raise ValueError(
            f"MEA {mea} has no DMD geometry (size_dmd / pxl_size_dmd) in params.rig_params."
        )
    mea_um = params.n_electrodes * params.mea_spacing
    dmd_x_um = size_dmd[0] * pxl
    return mea_um / dmd_x_um


# Geometry of each square along its own axis, as multiples of the MEA half-extent:
#   outer edge = (1 + SQUARE_DEPTH_FACTOR) * half-extent  (how far it reaches outward)
#   inner edge = (1 - SQUARE_INSET_FACTOR) * half-extent  (how far it comes INTO the MEA)
# With inset 0 the square just abuts the MEA edge; 0.5 brings its inner edge halfway to the
# MEA centre, so the cells of the outer half of the MEA see it (few RFs lie outside the
# MEA, which is why abutting squares gave too few responsive cells). 1 would reach the
# centre. Along the perpendicular axis a square always spans the MEA's full width.
SQUARE_DEPTH_FACTOR = 1.5
SQUARE_INSET_FACTOR = 0.5


def four_squares_bounds(
    fraction, depth_factor=SQUARE_DEPTH_FACTOR, inset_factor=SQUARE_INSET_FACTOR
):
    """Normalised [-1, 1] bounds ``(x0, x1, y0, y1)`` of each square, in FOUR_SQUARES order.

    ``fraction`` is the MEA half-extent (mea_extent_on_display). Each square runs from
    ``(1 - inset_factor) * fraction`` to ``(1 + depth_factor) * fraction`` along its own
    axis (so it overlaps the outer part of the MEA and reaches beyond it), and spans the
    MEA's width along the parallel axis.
    """
    f = fraction
    inner = (1 - inset_factor) * f
    outer = (1 + depth_factor) * f
    return [
        ("top", (-f, f, inner, outer)),
        ("right", (inner, outer, -f, f)),
        ("bottom", (-f, f, -outer, -inner)),
        ("left", (-outer, -inner, -f, f)),
    ]


def make_four_squares_frames(size, fraction):
    """Build the four black frames, each with one white square (FOUR_SQUARES order).

    Args:
        size: frame side length in pixels (square frame).
        fraction: MEA half-extent as a fraction of the display (mea_extent_on_display).

    Returns:
        list of (name, frame) — frame is a float array in {0, 1}, shape (size, size), with
        +y up so the "top" square is at the top of the displayed image.
    """
    frames = []
    for name, (x0, x1, y0, y1) in four_squares_bounds(fraction):
        frame = np.zeros((size, size))
        col0 = int(round((x0 + 1) / 2 * size))
        col1 = int(round((x1 + 1) / 2 * size))
        row0 = int(round((1 - y1) / 2 * size))  # +y up -> higher y is a lower row index
        row1 = int(round((1 - y0) / 2 * size))
        frame[row0:row1, col0:col1] = 1.0
        frames.append((name, frame))
    return frames


def standard_bin_frames(size, mea=None):
    """The six standard leading frames every stimulus bin should start with.

    Returns [grey, black, top, right, bottom, left] (indices STANDARD_BIN_FRAME_INDEX).
    Prepend these to your stimulus frames so the bin always carries the mean-luminance grey,
    the black gap frame and the four spatial-check squares. Your real stimulus frames then
    start at index N_STANDARD_BIN_FRAMES (6).
    """
    fraction = mea_extent_on_display(mea)
    grey = np.full((size, size), 0.5)
    black = np.zeros((size, size))
    squares = dict(make_four_squares_frames(size, fraction))
    return [grey, black] + [squares[name] for name, _ in FOUR_SQUARES]


def standard_preamble_vec_rows(
    refresh_hz=40,
    square_duration_s=1.0,
    gap_duration_s=1.0,
    n_square_reps=1,
    n_digit_for_rep=4,
):
    """Vec rows for the standard preamble: the four squares in turn, black in between.

    Starts with ``gap_duration_s`` of black under key 0 (ignored by the analysis), then for
    each square: ``square_duration_s`` of the white square followed by ``gap_duration_s``
    of black, BOTH under the square's key. The black that follows a square belongs to it,
    so a cell's OFF response is credited to the right square and the next square starts
    from black, with no response from the previous one still running.

    Uses the standard bin frame layout and the reserved sequence ids, so these rows can be
    PREPENDED to any real stimulus's vec rows without clashing. (The real stimulus keeps its
    own 1..N ids, but its frame indices must be offset by N_STANDARD_BIN_FRAMES since the
    six standard frames come first in the bin.)

    ``n_digit_for_rep`` must be the repetition-digit width of the stimulus these rows are
    prepended to, so that one decode works for the whole vec.

    Returns an (n, 5) int array of rows [phasemask, frame_index, color, shutter, key].
    """
    black = STANDARD_BIN_FRAME_INDEX["black"]
    n_square, n_gap = (
        round(square_duration_s * refresh_hz),
        round(gap_duration_s * refresh_hz),
    )
    rows = [[0, black, 0, 0, 0]] * n_gap  # leading black, key 0 = ignored
    for rep in range(n_square_reps):
        for name, _ in FOUR_SQUARES:
            key = FOUR_SQUARES_SEQUENCE_IDS[name] * 10**n_digit_for_rep + rep
            rows += [[0, STANDARD_BIN_FRAME_INDEX[name], 0, 0, key]] * n_square
            rows += [[0, black, 0, 0, key]] * n_gap
    return np.array(rows, dtype=int)


def prepend_standard_preamble(
    stimulus_bin_path,
    stimulus_vec_path,
    output_bin_path,
    output_vec_path,
    mea=None,
    refresh_hz=40,
    square_duration_s=1.0,
    gap_duration_s=1.0,
    n_square_reps=1,
    n_digit_for_rep=None,
):
    """Add the standard preamble (grey, four squares) to an existing stimulus bin + vec.

    This is the one-call way to make any stimulus follow the convention. It reads your
    stimulus's ``.bin`` and ``_std.vec`` and writes NEW files that:

      * start with the six standard frames (grey, black, top, right, bottom, left), then
        your stimulus's frames — copied byte-for-byte so they are unchanged;
      * play the four squares with black in between (reserved ids 9991-9994) before your
        stimulus's vec rows, whose frame indices are shifted by six (the standard frames
        come first) and whose own sequence ids are left untouched.

    The stimulus bin must be for the same rig ``mea`` it will be displayed on (its frames are
    copied as-is), and square (as the four-squares geometry assumes).

    Args:
        stimulus_bin_path, stimulus_vec_path: your existing stimulus files.
        output_bin_path, output_vec_path: where to write the preamble-prefixed files.
        mea: rig id (defaults to params.MEA).
        refresh_hz, square_duration_s, gap_duration_s, n_square_reps: preamble timing
            (see standard_preamble_vec_rows).
        n_digit_for_rep: repetition-digit width of the stimulus's keys. The preamble keys
            are written with the same width so one decode works for the whole vec. None
            (default) infers it from the stimulus vec (utils.infer_rep_digits) and prints
            it — check it against how you built your keys.
    """
    from .binfile import BinFile
    from .vec import infer_rep_digits

    mea = params.MEA if mea is None else mea

    header = BinFile.read_header(stimulus_bin_path)
    size = header["xsize"]
    if header["xsize"] != header["ysize"]:
        raise ValueError(
            f"prepend_standard_preamble expects a square stimulus bin, got "
            f"{header['xsize']}x{header['ysize']}."
        )

    # Read the stimulus frames as RAW bytes so they are copied exactly (no decode/re-encode).
    reader = BinFile(stimulus_bin_path, 0, 0, mea, mode="r")
    stimulus_frame_bytes = [reader.read_frame_as_bytes(i) for i in range(len(reader))]
    reader.close()

    stimulus_rows = np.loadtxt(stimulus_vec_path)[1:].astype(
        int
    )  # drop the summary row
    if n_digit_for_rep is None:
        n_digit_for_rep = infer_rep_digits(stimulus_rows[:, -1])
        print(
            f"Stimulus keys look like they use {n_digit_for_rep} repetition digit(s); the "
            "preamble keys will use the same (pass n_digit_for_rep= to override)."
        )

    preamble_frames = standard_bin_frames(size, mea=mea)
    preamble_rows = standard_preamble_vec_rows(
        refresh_hz=refresh_hz,
        square_duration_s=square_duration_s,
        gap_duration_s=gap_duration_s,
        n_square_reps=n_square_reps,
        n_digit_for_rep=n_digit_for_rep,
    )

    # vec: the stimulus's frame indices shift by the number of standard frames prepended.
    shifted_rows = stimulus_rows.copy()
    shifted_rows[:, 1] += N_STANDARD_BIN_FRAMES
    out_rows = np.vstack([preamble_rows, shifted_rows])
    out_header = [0, len(out_rows), 0, 0, 0]

    # bin: standard frames (encoded fresh) then the stimulus frames (raw byte copy).
    writer = BinFile(
        output_bin_path,
        size,
        size,
        mea,
        nb_images=len(preamble_frames) + len(stimulus_frame_bytes),
        mode="w",
    )
    for frame in preamble_frames:
        writer.append(frame)
    for frame_bytes in stimulus_frame_bytes:
        writer.append(frame_bytes)
    writer.close()

    np.savetxt(output_vec_path, np.vstack([out_header, out_rows]).astype(int), fmt="%d")
    print(
        f"Wrote {output_bin_path}\n      {output_vec_path}\n"
        f"  preamble: {len(preamble_frames)} frames + {len(preamble_rows)} vec rows "
        f"-> your stimulus frames now start at index {N_STANDARD_BIN_FRAMES}."
    )
    # Verify the written pair by content (the vec's square keys must point at the squares).
    check_standard_preamble(
        output_bin_path, output_vec_path, mea=mea, n_digit_for_rep=n_digit_for_rep
    )


def check_standard_preamble(
    bin_path, vec_path, mea=None, n_digit_for_rep=None, show=True
):
    """Verify, by CONTENT, that a vec's preamble rows point at the right frames of its bin.

    This does not trust any assumed frame offset. For every square key found in the vec
    (9991-9994) it reads the frame that key points to from the bin and compares it with the
    square that key is supposed to show; it also finds where the standard frames really are
    in the bin and checks that the stimulus's own rows all start after them. A figure shows
    the frame each square key points at, so you can confirm by eye: top square must be at
    the top, etc.

    Prints a report and returns True if everything matches, False otherwise. Run it on any
    (bin, vec) pair before displaying it on the rig. ``n_digit_for_rep`` is the repetition-
    digit width of the vec's keys (None = infer it from the vec).
    """
    from .binfile import BinFile
    from .vec import infer_rep_digits

    mea = params.MEA if mea is None else mea
    rows = np.loadtxt(vec_path)[1:].astype(int)
    frame_col, key_col = rows[:, 1], rows[:, 4]
    if n_digit_for_rep is None:
        n_digit_for_rep = infer_rep_digits(key_col)
    sequence_ids = key_col // 10**n_digit_for_rep

    reader = BinFile(bin_path, 0, 0, mea, mode="r")
    nb_frames = len(reader)
    expected = dict(make_four_squares_frames(reader.xsize, mea_extent_on_display(mea)))
    ok = True
    print(f"Checking {vec_path}\n against  {bin_path}  ({nb_frames} frames)")

    # 1) Where are the standard frames in this bin, really? (first frames only)
    found_at = {}
    for i in range(min(nb_frames, 3 * N_STANDARD_BIN_FRAMES)):
        frame = reader.read_frame(i)
        for name, square in expected.items():
            if name not in found_at and np.mean(np.abs(frame - square)) < 0.01:
                found_at[name] = i
    if len(found_at) < len(expected):
        missing = sorted(set(expected) - set(found_at))
        print(f"  x squares not found in the bin's first frames: {missing}")
        ok = False
        n_leading = N_STANDARD_BIN_FRAMES
    else:
        n_leading = max(found_at.values()) + 1
        print(
            f"  squares found at frames {found_at} -> stimulus frames start at {n_leading}"
        )
        if n_leading != N_STANDARD_BIN_FRAMES:
            print(
                f"  ! this bin has {n_leading} leading frames, the current convention has "
                f"{N_STANDARD_BIN_FRAMES} (old layout?). Fine as long as the vec agrees, "
                "checked below."
            )

    # 2) Does each square key of the vec point at the matching square frame?
    pointed = {}
    for name, seq_id in FOUR_SQUARES_SEQUENCE_IDS.items():
        indices = np.unique(frame_col[sequence_ids == seq_id])
        if len(indices) == 0:
            print(f"  - no rows with key {seq_id} ({name} square) in the vec")
            continue
        # A square's key covers the white square and the black gap after it: keep the
        # non-black frame(s) to identify the square.
        non_black = [
            int(i)
            for i in indices
            if not (0 <= i < nb_frames and reader.read_frame(int(i)).max() == 0)
        ]
        if len(non_black) != 1:
            print(
                f"  x key {seq_id} ({name}) points at several non-black frames: {non_black}"
            )
            ok = False
            if not non_black:
                continue
        index = non_black[0]
        if not 0 <= index < nb_frames:
            print(f"  x key {seq_id} ({name}) points at frame {index}, out of the bin")
            ok = False
            continue
        frame = reader.read_frame(index)
        pointed[name] = (index, frame)
        if np.mean(np.abs(frame - expected[name])) < 0.01:
            print(
                f"  v key {seq_id} ({name:6s}) -> frame {index}: matches the {name} square"
            )
        else:
            print(
                f"  x key {seq_id} ({name:6s}) -> frame {index}: is NOT the {name} square"
            )
            ok = False

    # 3) Do the stimulus's own rows point after the standard frames, and inside the bin?
    own = frame_col[(sequence_ids < 9990) & (key_col != 0)]  # key 0 = ignored rows
    if len(own):
        if own.min() < n_leading:
            print(
                f"  x stimulus rows point at frame {own.min()} < {n_leading}: frame indices "
                "were not offset (or offset by the wrong amount)"
            )
            ok = False
        if own.max() >= nb_frames:
            print(
                f"  x stimulus rows point at frame {own.max()} >= {nb_frames} frames in bin"
            )
            ok = False
        if own.min() >= n_leading and own.max() < nb_frames:
            print(
                f"  v stimulus rows use frames {own.min()}..{own.max()} (inside the bin)"
            )
    reader.close()

    print(
        "  => PREAMBLE OK" if ok else "  => PREAMBLE PROBLEM, do not display this pair"
    )

    if show and pointed:
        fig, axs = plt.subplots(1, len(pointed), figsize=(4 * len(pointed), 4))
        for ax, (name, (index, frame)) in zip(np.atleast_1d(axs), pointed.items()):
            ax.imshow(frame, cmap="gray", vmin=0, vmax=1)
            ax.set_title(
                f"key {FOUR_SQUARES_SEQUENCE_IDS[name]} -> frame {index}\nshould be {name}",
                fontsize=13,
            )
            ax.axis("off")
        fig.suptitle("Frame each square key points at (check by eye)", fontsize=14)
        plt.show()
    return ok


def _rf_center(sta_analysis):
    """RF centre of one cell from its STA, in STA pixels, centred with +x right, +y up."""
    ny, nx = np.asarray(sta_analysis["Spatial"]).shape
    _, x0, y0, *_ = sta_analysis["EllipseCoor"]
    return x0 - nx / 2, ny / 2 - y0


def _mean_spikes_per_rep(sequence):
    """Mean number of spikes per repetition during a sequence (its response strength)."""
    raster = sequence["raster"]
    if not raster:
        return 0.0
    return float(np.mean([len(rep) for rep in raster]))


def plot_four_squares_response_maps(
    spikes_per_sequence_dict,
    sta_results,
    square_keys=FOUR_SQUARES_KEYS,
    mea=None,
    min_spikes: float = 2.0,
    selectivity: float = 2.0,
    smoothing_um: float = 100.0,
    min_cells: float = 2.0,
    dot_diameter_um: float = 200.0,
    statistic: str = "preference",
    fontsize: int = 14,
    cmap: str = "inferno",
):
    """Map of which square the cells prefer, over the retina.

    One panel per square (FOUR_SQUARES order), in micrometres centred on the MEA. Each cell
    is a circle of about receptive-field size (``dot_diameter_um``) at its RF centre
    (checkerboard STA), so you see the retina actually covered.

    The question is "which square does each region of the retina respond to", so the default
    statistic is a VOTE, not an average: a cell is *selective* if its best square evoked at
    least ``min_spikes`` per presentation AND at least ``selectivity`` times its second-best
    square; each selective cell votes for its best square. The colour at any point of panel
    k is the fraction of the selective cells around it (Gaussian kernel, sigma =
    ``smoothing_um``) that voted for square k. Cells that respond to nothing or to
    everything alike do not vote and do not dilute the map (they are drawn dimmer). Regions
    resting on fewer than ``min_cells`` voters stay black. The check: next to the square
    that was on (cyan outline) the fraction should be high; the MEA footprint is dashed.

    ``statistic="mean"`` gives the older view instead: the kernel-averaged response
    (normalised per cell) of every cell passing ``min_spikes``.

    Works on any recording whose vec starts with the standard preamble, as well as on a
    standalone four-squares recording. Needs the STA analysis in physical units
    (``Spatial_unit_size_um`` in the STA results) to place cells and squares in the same
    frame.

    Args:
        spikes_per_sequence_dict: output of build_spikes_per_sequence_dict for a recording
            whose sequence keys include the four squares.
        sta_results: the checkerboard STA results (``sta_data_analysed_extended.pkl``),
            ``{cell_id: {"sta_analysis": {...}}}`` — provides each cell's RF.
        square_keys: the four sequence keys, in FOUR_SQUARES order (top, right, bottom, left).
            Defaults to the reserved ids (FOUR_SQUARES_KEYS).
        mea: rig id (defaults to params.MEA), for the square / MEA geometry.
        min_spikes: a cell counts only if its best square evoked at least this many spikes
            per presentation on average. 0 keeps every cell.
        selectivity: ("preference" only) best / second-best ratio a cell must reach to be
            considered selective and vote. 1 lets every responsive cell vote.
        smoothing_um: sigma of the Gaussian kernel, in µm. Larger = smoother map.
        min_cells: minimum kernel weight (≈ number of nearby voting cells; a cell right at
            the point weighs 1, one sigma away 0.6) below which the map is left black.
        dot_diameter_um: diameter of each cell's circle, in µm.
        statistic: "preference" (default, the vote described above) or "mean" (kernel-
            averaged per-cell-normalised response of all responsive cells).
        fontsize: base font size.
        cmap: colormap for the response.

    Returns:
        The matplotlib Figure.
    """
    mea = params.MEA if mea is None else mea
    names = [name for name, _ in FOUR_SQUARES]

    # Cells present in both the responses and the STA, that have all four squares.
    cells = [
        c
        for c in spikes_per_sequence_dict
        if c in sta_results
        and all(k in spikes_per_sequence_dict[c] for k in square_keys)
    ]
    if not cells:
        some_cell = next(iter(spikes_per_sequence_dict), None)
        found_keys = sorted(spikes_per_sequence_dict.get(some_cell, {}))
        raise ValueError(
            f"No cell has both the four squares {list(square_keys)} and an STA.\n"
            f"  sequence keys of this recording: {found_keys}\n"
            "  - if the square keys are missing, this vec has no standard preamble, or "
            "n_digit_for_rep does not match the width used when the preamble was written;\n"
            "  - if they are there, check the STA (checkerboard) analysis was run for this "
            "experiment and that the cell ids match."
        )
    n_presentations = len(spikes_per_sequence_dict[cells[0]][square_keys[0]]["raster"])

    # RF centres in µm, centred on the MEA (+x right, +y up), and responses per square.
    xs, ys, responses = [], [], []
    for cell in cells:
        sta_analysis = sta_results[cell]["sta_analysis"]
        px_um = sta_analysis.get("Spatial_unit_size_um")
        if not px_um:
            raise ValueError(
                f"Cell {cell} has no 'Spatial_unit_size_um' in its STA results: run the "
                "physical-units step of the checkerboard analysis (notebook 2) so RF "
                "positions and squares can be drawn in the same frame."
            )
        cx, cy = _rf_center(sta_analysis)
        xs.append(cx * px_um)
        ys.append(cy * px_um)
        responses.append(
            [
                _mean_spikes_per_rep(spikes_per_sequence_dict[cell][k])
                for k in square_keys
            ]
        )
    xs, ys = np.array(xs), np.array(ys)
    responses = np.array(responses)  # (n_cells, 4)

    if statistic not in ("preference", "mean"):
        raise ValueError(f"statistic must be 'preference' or 'mean', got {statistic!r}")

    # Responsive = the best square evokes at least min_spikes. Selective = on top of that,
    # the best square clearly beats the second-best. Only these cells say something about
    # WHICH square a region responds to.
    order = np.sort(responses, axis=1)
    peak, second = order[:, -1], order[:, -2]
    responsive = peak >= min_spikes
    selective = responsive & (peak >= selectivity * np.maximum(second, 1e-9))
    preferred = responses.argmax(axis=1)
    n_silent = int((~responsive).sum())

    if statistic == "preference":
        voters = selective
        # One-hot vote of each selective cell: kernel-averaging it gives, per square, the
        # fraction of the nearby selective cells that prefer that square.
        values = np.eye(4)[preferred]
        color_label = "fraction of nearby selective cells preferring this square"
        subtitle = (
            f"{int(selective.sum())} selective cells vote, "
            f"{int((responsive & ~selective).sum())} respond but do not discriminate, "
            f"{n_silent} silent"
        )
    else:
        voters = responsive
        values = responses / np.where(peak > 0, peak, 1)[:, None]  # best square -> 1
        color_label = "response (normalised per cell, kernel-averaged)"
        subtitle = (
            f"{int(responsive.sum())} responsive cells, {n_silent} silent dropped"
        )
    vmin, vmax = 0.0, 1.0
    if not voters.any():
        raise ValueError(
            "No cell passes the criteria (min_spikes / selectivity): lower them, or check "
            "the squares were actually displayed."
        )

    # Geometry of the squares and the MEA, in µm (same frame as the RF centres).
    rig = params.get_rig_params(mea)
    half_display_um = rig["size_dmd"][0] * rig["pxl_size_dmd"] / 2
    fraction = mea_extent_on_display(mea)
    mea_half_um = fraction * half_display_um
    bounds_um = {
        name: tuple(v * half_display_um for v in b)
        for name, b in four_squares_bounds(fraction)
    }
    reach_um = max(abs(v) for b in bounds_um.values() for v in b)
    extent = max(reach_um, np.abs(xs).max(), np.abs(ys).max()) * 1.05

    # Kernel-smoothed map over the VOTING cells only: at each grid point, the Gaussian-
    # weighted mean of their values. Points resting on fewer than min_cells cells' worth of
    # weight stay black, so no region of the map hangs on a single cell.
    vx, vy, vvalues = xs[voters], ys[voters], values[voters]
    n_grid = 120
    grid = np.linspace(-extent, extent, n_grid)
    gx, gy = np.meshgrid(grid, grid)  # gy increases with row index (+y up)
    d2 = (gx[..., None] - vx) ** 2 + (gy[..., None] - vy) ** 2  # (grid, grid, n_voters)
    weights = np.exp(-d2 / (2 * smoothing_um**2))
    total_weight = weights.sum(axis=2)
    smoothed = np.tensordot(weights, vvalues, axes=([2], [0]))  # (n_grid, n_grid, 4)
    with np.errstate(invalid="ignore", divide="ignore"):
        smoothed = smoothed / total_weight[..., None]
    smoothed[total_weight < min_cells] = np.nan

    colormap = plt.get_cmap(cmap).copy()
    colormap.set_bad("black")  # no cell nearby
    fig, axs = plt.subplots(1, 4, figsize=(20, 5.5), facecolor="black")
    image = None
    for square_index, (ax, name) in enumerate(zip(axs, names)):
        ax.set_facecolor("black")
        image = ax.imshow(
            np.ma.masked_invalid(smoothed[:, :, square_index]),
            cmap=colormap,
            vmin=vmin,
            vmax=vmax,
            origin="lower",  # row index grows with +y, so put row 0 at the bottom
            extent=[-extent, extent, -extent, extent],
            interpolation="bilinear",
        )
        # The cells themselves, drawn at about receptive-field size (in data units, so the
        # size is physical): voting cells in white, the others dimmer (coverage only).
        for x, y, votes in zip(xs, ys, voters):
            ax.add_patch(
                plt.Circle(
                    (x, y),
                    dot_diameter_um / 2,
                    fill=False,
                    edgecolor="white" if votes else "0.5",
                    alpha=0.35 if votes else 0.2,
                    lw=0.6,
                )
            )
        # MEA footprint (grey) and the square that was on (cyan).
        ax.add_patch(
            plt.Rectangle(
                (-mea_half_um, -mea_half_um),
                2 * mea_half_um,
                2 * mea_half_um,
                fill=False,
                edgecolor="0.5",
                lw=1.2,
                ls="--",
            )
        )
        x0, x1, y0, y1 = bounds_um[name]
        ax.add_patch(
            plt.Rectangle(
                (x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="tab:cyan", lw=2.5
            )
        )
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
        ax.set_aspect("equal")
        ax.set_title(f"{name} square", fontsize=fontsize, color="white")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("0.3")
    axs[0].set_ylabel(
        "RF position, µm from MEA centre (+y = up)",
        fontsize=fontsize - 1,
        color="white",
    )
    cbar = fig.colorbar(image, ax=axs, fraction=0.015, pad=0.01)
    cbar.set_label(color_label, fontsize=fontsize - 1, color="white")
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")
    fig.suptitle(
        f"Which square does each region of the retina respond to?  ({subtitle}; "
        f"{n_presentations} presentation(s) per square, kernel σ = {smoothing_um:.0f} µm)",
        fontsize=fontsize + 2,
        color="white",
    )
    return fig
