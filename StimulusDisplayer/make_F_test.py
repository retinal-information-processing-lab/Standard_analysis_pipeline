"""Generate a standard "F" orientation-test stimulus (bin + vec) for the displayer.

The letter F is strongly asymmetric, so it makes any unwanted flip or rotation of the
display obvious. This is a test of the RIG (display chain); it does not replace the F you
build with your own stimulus code (see B_Standard_Stim_Creation, section II).

Run from the StimulusDisplayer folder:  python make_F_test.py
It writes BIN/F_TEST_432x432.bin and VEC/F_TEST_432x432.vec .
"""

import os
import sys

import numpy as np

# The displayer lives inside the pipeline repository and shares its BinFile and rig table.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import params  # noqa: E402
from utils.binfile import BinFile  # noqa: E402

# ---- F canvas ----------------------------------------------------------------
SIZE = 432  # square window, in pixels (matches the 432 x 432 stimuli)
MARGIN = 90  # blank border around the F, in pixels
THICK = 55  # stroke thickness, in pixels
RIG_ID = 2  # MEA/rig id used to write the bin (2 -> DMD up to 1920 x 1080)
N_FRAMES = 80  # number of vec rows (~2 s at 40 Hz); the F is static so the count only sets duration


def make_F() -> np.ndarray:
    """A white upright 'F' on a black 432 x 432 background, as floats in [0, 1]."""
    frame = np.zeros((SIZE, SIZE), dtype=float)
    frame[MARGIN : SIZE - MARGIN, MARGIN : MARGIN + THICK] = (
        1.0  # vertical stroke (left)
    )
    frame[MARGIN : MARGIN + THICK, MARGIN : SIZE - MARGIN] = (
        1.0  # top stroke (full width)
    )
    mid = SIZE // 2
    frame[mid - THICK // 2 : mid + THICK // 2, MARGIN : SIZE - MARGIN - 50] = (
        1.0  # middle stroke (shorter)
    )
    return frame


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(here, "BIN"), exist_ok=True)
    os.makedirs(os.path.join(here, "VEC"), exist_ok=True)
    bin_path = os.path.join(here, "BIN", "F_TEST_432x432.bin")
    vec_path = os.path.join(here, "VEC", "F_TEST_432x432.vec")

    # ---- bin: frame 0 = black, frame 1 = F (F is the second frame by convention) ----
    binf = BinFile(
        bin_path,
        SIZE,
        SIZE,
        RIG_ID,
        nb_images=2,
        mode="w",
        rig_settings=params.get_display_rig_params(RIG_ID),
    )
    binf.append(np.zeros((SIZE, SIZE), dtype=float))
    binf.append(make_F())
    binf.close()

    # ---- vec: header row (col 1 = total frames) then N rows showing frame index 1 ----
    # columns: phasemask-switch, frame index, color, shutter, sequence key (seq 0, rep 0)
    header = f"0 {N_FRAMES} 0 0 0"
    rows = ["0 1 0 0 0"] * N_FRAMES
    with open(vec_path, "w") as f:
        f.write("\n".join([header] + rows) + "\n")

    print(f"wrote {bin_path}")
    print(f"wrote {vec_path}")


if __name__ == "__main__":
    main()
