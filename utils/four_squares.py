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
#     1 : F      (orientation test — asymmetric, reveals any flip/rotation)
#     2 : top square    3 : right square    4 : bottom square    5 : left square
#     6.. : the real stimulus frames
#
#   Standard vec: show F briefly, then the four squares, THEN the real stimulus.
#
# The preamble uses RESERVED sequence ids in the 999x block so they can never collide with a
# real stimulus's sequence ids (which start at 1 and rarely exceed ~100). Keys are still
# ``sequence_id * 10000 + repetition`` (4 repetition digits), e.g. F -> 99990000, top square
# rep 0 -> 99910000. A real stimulus keeps its own 1..N ids alongside, untouched.
F_CHECK_SEQUENCE_ID = 9999
FOUR_SQUARES_SEQUENCE_IDS = {"top": 9991, "right": 9992, "bottom": 9993, "left": 9994}

# The four squares' sequence keys as the analysis sees them (the id part of the vec key),
# in canonical order — the default the response plot expects.
FOUR_SQUARES_KEYS = [str(FOUR_SQUARES_SEQUENCE_IDS[name]) for name, _ in FOUR_SQUARES]

# Frame index of each standard leading frame in the bin.
STANDARD_BIN_FRAME_INDEX = {
    "grey": 0,
    "F": 1,
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


# How far each square reaches OUTWARD from the MEA edge, as a multiple of the MEA half-
# extent. The inner edge always stays flush with the MEA (so the squares never overlap it);
# a bigger factor just makes the bright rectangles reach further out. 1.0 abuts by exactly
# the MEA half-size; 1.5 makes them 1.5x deeper.
SQUARE_DEPTH_FACTOR = 1.5


def four_squares_bounds(fraction, depth_factor=SQUARE_DEPTH_FACTOR):
    """Normalised [-1, 1] bounds ``(x0, x1, y0, y1)`` of each square, in FOUR_SQUARES order.

    ``fraction`` is the MEA half-extent (mea_extent_on_display). Each square abuts one MEA
    edge, extends outward by ``depth_factor * fraction``, and spans the MEA's width along the
    parallel axis. The inner edge stays on the MEA edge, so widening (a larger depth_factor)
    never oversteps the MEA.
    """
    f = fraction
    d = depth_factor * f  # outward depth of each square
    return [
        ("top", (-f, f, f, f + d)),
        ("right", (f, f + d, -f, f)),
        ("bottom", (-f, f, -f - d, -f)),
        ("left", (-f - d, -f, -f, f)),
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


def make_letter_f(size):
    """A white upright letter 'F' on black, shape (size, size), float in {0, 1}.

    The F is strongly asymmetric, so any unwanted flip or rotation of the display is obvious.
    """
    margin, thick = size // 5, size // 8
    frame = np.zeros((size, size))
    frame[margin : size - margin, margin : margin + thick] = 1.0  # vertical stroke
    frame[margin : margin + thick, margin : size - margin] = 1.0  # top stroke
    mid = size // 2
    frame[mid - thick // 2 : mid + thick // 2, margin : size - margin - thick] = 1.0
    return frame


def standard_bin_frames(size, mea=None):
    """The six standard leading frames every stimulus bin should start with.

    Returns [grey, F, top, right, bottom, left] (indices STANDARD_BIN_FRAME_INDEX). Prepend
    these to your stimulus frames so the bin always carries the mean-luminance grey, the F
    orientation test, and the four spatial-check squares. Your real stimulus frames then
    start at index N_STANDARD_BIN_FRAMES (6).
    """
    fraction = mea_extent_on_display(mea)
    grey = np.full((size, size), 0.5)
    squares = dict(make_four_squares_frames(size, fraction))
    return [grey, make_letter_f(size)] + [squares[name] for name, _ in FOUR_SQUARES]


def standard_preamble_vec_rows(
    refresh_hz=40,
    f_duration_s=0.5,
    square_duration_s=1.0,
    n_square_reps=1,
):
    """Vec rows for the standard preamble: show the F briefly, then each of the four squares.

    Uses the standard bin frame layout (F = 1, squares = 2..5) and the reserved sequence ids,
    so these rows can be PREPENDED to any real stimulus's vec rows without clashing. (The real
    stimulus keeps its own 1..N ids, but its frame indices must be offset by
    N_STANDARD_BIN_FRAMES since the six standard frames come first in the bin.)

    Returns an (n, 5) int array of rows [phasemask, frame_index, color, shutter, key].
    """
    rows = []
    for _ in range(round(f_duration_s * refresh_hz)):
        rows.append(
            [0, STANDARD_BIN_FRAME_INDEX["F"], 0, 0, F_CHECK_SEQUENCE_ID * 10000]
        )
    for rep in range(n_square_reps):
        for name, _ in FOUR_SQUARES:
            frame_index = STANDARD_BIN_FRAME_INDEX[name]
            key = FOUR_SQUARES_SEQUENCE_IDS[name] * 10000 + rep
            for _ in range(round(square_duration_s * refresh_hz)):
                rows.append([0, frame_index, 0, 0, key])
    return np.array(rows, dtype=int)


def prepend_standard_preamble(
    stimulus_bin_path,
    stimulus_vec_path,
    output_bin_path,
    output_vec_path,
    mea=None,
    refresh_hz=40,
    f_duration_s=0.5,
    square_duration_s=1.0,
    n_square_reps=1,
):
    """Add the standard preamble (grey, F, four squares) to an existing stimulus bin + vec.

    This is the one-call way to make any stimulus follow the convention. It reads your
    stimulus's ``.bin`` and ``_std.vec`` and writes NEW files that:

      * start with the six standard frames (grey, F, top, right, bottom, left), then your
        stimulus's frames — copied byte-for-byte so they are unchanged;
      * play the F briefly then the four squares (reserved ids 9999 / 9991-9994) before your
        stimulus's vec rows, whose frame indices are shifted by six (the standard frames come
        first) and whose own sequence ids are left untouched.

    The stimulus bin must be for the same rig ``mea`` it will be displayed on (its frames are
    copied as-is), and square (as the four-squares geometry assumes).

    Args:
        stimulus_bin_path, stimulus_vec_path: your existing stimulus files.
        output_bin_path, output_vec_path: where to write the preamble-prefixed files.
        mea: rig id (defaults to params.MEA).
        refresh_hz, f_duration_s, square_duration_s, n_square_reps: preamble timing
            (see standard_preamble_vec_rows).
    """
    from .binfile import BinFile

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

    preamble_frames = standard_bin_frames(size, mea=mea)
    preamble_rows = standard_preamble_vec_rows(
        refresh_hz=refresh_hz,
        f_duration_s=f_duration_s,
        square_duration_s=square_duration_s,
        n_square_reps=n_square_reps,
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
    normalize: bool = True,
    fontsize: int = 14,
    cmap: str = "inferno",
):
    """Plot each cell's RF position coloured by its response to each of the four squares.

    One subplot per square (in FOUR_SQUARES order): every cell is drawn at its RF centre
    (from the checkerboard STA) and coloured by how much it fired during that square. All
    four share one colour scale, and an arrow points toward the square that was on. The
    check: in the "top" subplot the top cells should be brightest, and so on.

    Args:
        spikes_per_sequence_dict: output of build_spikes_per_sequence_dict for the
            four-squares recording (its sequence keys include the four squares).
        sta_results: the checkerboard STA results (``sta_data_analysed_extended.pkl``),
            ``{cell_id: {"sta_analysis": {...}}}`` — provides each cell's RF.
        square_keys: the four sequence keys, in FOUR_SQUARES order (top, right, bottom, left).
            Defaults to the reserved ids (FOUR_SQUARES_KEYS).
        normalize: if True (default), each cell's four responses are divided by that cell's
            strongest square, so the colour shows WHICH square each cell prefers rather than
            its absolute firing rate (a cell that fires the same to all four looks uniformly
            bright; a non-responsive cell stays dark). If False, colour is the absolute mean
            spikes per presentation, shared across the four panels.
        fontsize: base font size.
        cmap: colormap for the response.

    Returns:
        The matplotlib Figure.
    """
    names = [name for name, _ in FOUR_SQUARES]
    directions = dict(FOUR_SQUARES)

    # Cells present in both the responses and the STA, that have all four squares.
    cells = [
        c
        for c in spikes_per_sequence_dict
        if c in sta_results
        and all(k in spikes_per_sequence_dict[c] for k in square_keys)
    ]
    if not cells:
        raise ValueError(
            "No cell has both a four-squares response and an STA. Check that the STA "
            "(checkerboard) analysis was run for this experiment and that the cell ids match."
        )

    xs, ys, responses = [], [], []
    for cell in cells:
        cx, cy = _rf_center(sta_results[cell]["sta_analysis"])
        xs.append(cx)
        ys.append(cy)
        responses.append(
            [
                _mean_spikes_per_rep(spikes_per_sequence_dict[cell][k])
                for k in square_keys
            ]
        )
    xs, ys = np.array(xs), np.array(ys)
    responses = np.array(responses)  # (n_cells, 4)

    if normalize:
        # Per cell: divide the four responses by the cell's strongest square (0 if silent).
        peak = responses.max(axis=1, keepdims=True)
        responses = np.divide(
            responses, peak, out=np.zeros_like(responses), where=peak > 0
        )
        color_label = "response (normalised per cell)"
        vmin, vmax = 0.0, 1.0
    else:
        color_label = "mean spikes / presentation"
        vmin, vmax = float(responses.min()), float(responses.max())

    fig, axs = plt.subplots(1, 4, figsize=(20, 5.5))
    scatter = None
    for square_index, (ax, name) in enumerate(zip(axs, names)):
        scatter = ax.scatter(
            xs,
            ys,
            c=responses[:, square_index],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=70,
            edgecolor="k",
            linewidth=0.3,
        )
        ax.set_title(f"{name} square", fontsize=fontsize)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        # Arrow near the border pointing toward the square that was on.
        dx, dy = directions[name]
        ax.annotate(
            "",
            xytext=(0.5 + 0.42 * dx, 0.5 + 0.42 * dy),
            xy=(0.5 + 0.5 * dx, 0.5 + 0.5 * dy),
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="-|>", color="tab:cyan", lw=2.5),
        )
    axs[0].set_ylabel("RF position (+y = up)", fontsize=fontsize - 1)
    fig.colorbar(scatter, ax=axs, fraction=0.015, pad=0.01).set_label(
        color_label, fontsize=fontsize - 1
    )
    fig.suptitle(
        f"Receptive-field position vs response to each square  ({len(cells)} cells)",
        fontsize=fontsize + 2,
    )
    return fig
