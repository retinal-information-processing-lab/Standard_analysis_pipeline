# README — developers

This file is for people who want to **add or change code** in the pipeline. If you only want
to *use* it, read `README.md`.

The pipeline was renovated in three phases:

| phase | who | what |
|-------|-----|------|
| I (2025) | Ron, Steeve | first refactoring: notebooks split into cells + companion `.py` modules, conda env, first unit tests, `develop`/`main` branches |
| II (Jan–Feb 2026) | Chiara | checkerboard / STA analysis rewritten and standardised (one RF-fitting method, physical units, figure conventions) |
| III (Jun 2026 → ) | Baptiste | standard vec analysis, cell quality / typing, stimulus creation (preamble), utils package, tests + CI |

Change log at the bottom of this file. Roadmap just above it.

---

## 1. Setting up for development

1. Install the conda environment and register the Jupyter kernel exactly as in
   `README.md` (Installation). The environment runs **Python 3.9**: no `match`, no
   `X | None` annotations (that one crashes at import time), no `list[int]` outside quotes
   in code that runs at import.
2. Install the dev tool: `pip install ruff` (in the env).
3. Check everything works from the repository root:

```bash
python tasks.py test              # unit tests (stdlib unittest, no pytest needed)
python tasks.py check-formatting  # ruff lint + format check
python tasks.py fix-formatting    # ruff --fix + ruff format, run it before committing
```

Test dataset used for manual checks: `20251219_PulsingGratings_PupilSize/` (from Guilhem).
`params.py` is per-machine: it holds the experiment name, recording list and rig — you will
have your own local version, and it is normal for it to show as modified.

## 2. Branches and CI

- `main` is the released pipeline; `develop` is where work lands.
- Create a feature branch from `develop`, open a pull request back into `develop` when done.
- **CI** (`.github/workflows/test.yml`) runs `ruff check .` and `python -m unittest discover
  -s tests` on every push and pull request. There is no local git hook: run
  `python tasks.py fix-formatting` yourself before pushing, or CI will complain.
- `ruff.toml` holds the few rules we relax on purpose (star re-exports in `utils`, long lines,
  bare `except` in notebooks). Don't add per-file ignores without a reason in a comment.

Those developments tools are only in the develop branch not in main. 
Roadmap to push a change to main :
- develop in personal branch
- push changes to main develop branch and test
- push to main

## 3. How the code is organised

```
1-…5_ notebooks          the standard pipeline, in order (preprocessing → checkerboard →
                         DG → 4a cell quality → 4b cell typing → 5 ID card)
A_Standard_Vec_Analysis  generic raster/PSTH analysis of any stimulus described by a vec
B_Standard_Stim_Creation how to build a stimulus (bin + vec), the standard preamble
utils/                   all the Python code (see below)
params.py                the ONLY place for experiment / rig parameters
tests/                   unit tests (synthetic data, no recording needed)
ResourcesAndTools/       StandardVec (the vec files the analyses expect), StimMaking,
                         probe file, binary source of the checkerboard, RF tool, Typing
StimulusDisplayer/       stimulus preview tool (pygame GUI replaying a vec against a bin);
                         imports utils.binfile and params from the repo root
Other_Analysis_Notebooks/ non-standard analyses, not maintained to the same level
```

### `utils/` — shared vs companion modules

`utils` is a package. `utils/__init__.py` re-exports the **shared** modules
(`loading`, `preprocessing`, `checkerboard`, `sta`, `vec`, `reliability`, `cell_typing`,
`cell_quality`, `four_squares`, `RPV_analysis`, `binfile`, …), so notebooks call
`utils.load_obj(...)`, `utils.build_spikes_per_sequence_dict(...)`, etc.

Each standard notebook also has a **companion module** that is *not* re-exported and is
imported explicitly, e.g. `from utils import drifting_gratings as analysis`:

| notebook | companion |
|----------|-----------|
| 2-Analyse_Checkerboard | `utils/analyse_checkerboard.py` |
| 3-Drifting_Gratings | `utils/drifting_gratings.py` |
| 4a / 4b | `utils/chirp.py` (chirp rasters), `utils/cell_typing.py` |
| 5_Cell_ID_card | `utils/cell_id.py` |

Rule of thumb: code used by **two or more** notebooks goes in a shared module; code specific
to one notebook goes in its companion module; the notebook itself stays thin (inputs, one
call per step, a plot) with a few commented "escape hatch" examples showing how to do the
same thing by hand.

### Notebooks are for non-coders

Most users have little coding experience. In notebooks:

- every input a user may change is a named variable at the top of the cell, with a one-line
  comment saying what it does and a sensible default;
- one step per cell, a markdown cell before it saying *why*, not only *what*;
- no magic numbers in code: give them a name (`rpv_threshold = 0.5  # %`), and prefer
  module-level named constants in `utils` (`N_STANDARD_BIN_FRAMES`, `FOUR_SQUARES_KEYS`, …);
- figures: big fonts, few elements, readable from across the room. Clear outputs before
  committing a notebook (`git diff` on a notebook with outputs is unreadable).

### Data flow and files on disk

- `params.root/Analysis/output/` (= `params.output_directory`) holds everything the
  pipeline produces: `<exp>_fullexp_neurons_data.pkl` (spikes per cell per recording),
  `<exp>_cell_quality.pkl`, and one folder per analysis and recording:
  `Checkerboard_Analysis_rec_N`, `DG_Analysis_rec_N`, `CellTyping_Analysis_rec_N`,
  `Vec_Analysis_rec_N`. Find them with `utils.find_analysis_directory(output_directory, "DG")`.
- Triggers per recording: `params.triggers_directory/<exp>_<recording>_triggers.pkl`
  (`indices` in samples) and `_trigger_channels.pkl` (raw aux channels, for sanity checks).
- Vec files live in `params.stim_directory` (`ResourcesAndTools/StandardVec`). The analyses
  expect the `_std.vec` version: the stimulus vec with a **sequence key** in its last column
  (`<sequence id><repetition digits>`, see section I of notebook B).
- Results are plain dicts keyed by cell id, saved with `utils.save_obj` / `utils.load_obj`
  (pickle): `sta_data[cell]["sta_analysis"][...]`, `cell_quality[cell]["sta_ok"]`, … Keep
  new results in that shape — users index them directly.

### Conventions

- `snake_case` everywhere; docstrings on every public function (Args / Returns), written
  for someone who does not know the code.
- Pass what a function needs explicitly (`params.fs`, `params.MEA`), never the `params`
  module as a whole, and never read `params` from inside `utils` (except the rig table in
  `params.rig_params`, accessed through `params.get_rig_params(mea)`).
- Rig-dependent things (pixel size, DMD size, polarity, optical transform, trigger threshold)
  come **only** from `params.rig_params`. To support a new rig: fill in its entry and add
  it to `params.DISPLAY_READY_RIGS`.
- Images / STAs follow the `imshow` convention: row 0 at the top, x = columns, y = rows.
  Anything drawn on top of an STA (contours, ellipses, RF centres) must use the same
  orientation (see the cluster-figure bug in the change log).
- Sequence keys are strings of digits; the repetition number is the last `n_digit_for_rep`
  digits and is **inferred** from the vec (`utils.infer_rep_digits`) but always confirmed by
  the user. Don't hardcode `[-4:]` or `[-2:]`.
- Anything interactive (reviews, yes/no prompts) must be **resumable**: save after each
  decision, skip what was already decided on re-run.

## 4. Adding things

**A new standard stimulus analysis**

1. Build the stimulus with notebook B: vec with sequence keys, the standard preamble
   prepended (`utils.prepend_standard_preamble`), checked with `utils.check_standard_preamble`.
   Put the `_std.vec` in `ResourcesAndTools/StandardVec`.
2. Start from `A_Standard_Vec_Analysis`: `utils.build_spikes_per_sequence_dict` already gives
   you rasters, PSTHs and per-repetition triggers for any keyed vec. Only the
   stimulus-specific metrics and plots are new code.
3. Put those in a companion module `utils/<stimulus>.py`; keep the notebook thin.
4. Add a unit test in `tests/` with a tiny synthetic vec + spike train (see
   `tests/test_vec_sequences.py`) for anything that has logic (parsing, timing, metrics).
5. Add a line to `utils/__init__.py`'s docstring and to the change log below.

**A new quality criterion** — add a key to the per-cell dict built in notebook 4a
(`utils/cell_quality.py`, `QUALITY_CRITERIA`), evaluate it in its own 4a cell, save after
each decision.

**A shared helper** — put it in the matching shared module, docstring, test if it has logic,
and grep the notebooks for the duplicated code it replaces.

## 5. Tests

- Stdlib `unittest`, run with `python tasks.py test` or
  `python -m unittest tests.test_vec_sequences` for one file.
- Tests use **small synthetic inputs** (a handful of triggers and spikes) — no recording, no
  network. Regression tests on real data are out of scope: the figures are the visual check.
- `tests/test_utils.py` (loading helpers), `tests/test_vec_sequences.py` (the vec "sequence"
  machinery: key parsing, rasters, PSTH bins, repetition filtering),
  `tests/test_vec_sequences.py` also covers the repetition-range filter.
- When you fix a bug, add the test that would have caught it first (e.g. the 27/28-bin PSTH
  test was built from the real trigger samples that exposed it).

---

## Roadmap

- Analyses for the other standard stimuli: multisize spots, barcode, MSF.
- Remi's check with repeated stimuli (like the checkerboard) to accept / refuse merges
  during spike sorting.
- Notebook 5 (ID card) still recomputes the RPV itself; it should read `cell_quality`.
- SWN loader (`utils/analyse_checkerboard.load_swn_stimulus`) reads frame indices straight
  from the vec header: it does not support a preamble-prefixed vec (SWN is recorded without
  one for now).
- Also check issues on Git for more

## Change log

Newest first. Older entries summarised from the Phase I / II notes.

**2026-09-16 (Baptiste)** — four-squares preamble v2
- Preamble is now grey, **black**, four squares (6 frames, stimulus starts at 6). Each square
  = 1 s white + 1 s black under its own key, plus 1 s black (key 0) first, so OFF responses
  are credited to the right square. Squares moved **inward** (`SQUARE_INSET_FACTOR = 0.5`:
  inner edge halfway to the MEA centre) because too few RFs lie outside the MEA.
- Notebook A four-squares map rebuilt: µm frame centred on the MEA, kernel-smoothed
  **preference vote** of selective cells (silent / non-discriminating cells don't vote),
  black where < `min_cells`, RF-sized circles, cyan outline of the active square. Verified
  no y-axis reversal between RF centres and squares (round trip through BinFile).
- `infer_rep_digits` anchors on the preamble ids; loud warning when triggers ≠ vec rows;
  `find_vec_file` accepts a full path / empty name cleanly (notebook A vec selection).

**2026-09-15 (Baptiste)** — StimulusDisplayer merged into this repository
- The displayer was its own repo with a copy of `binfile.py` and a `rig_settings.py`
  mirroring `params.rig_params`; both copies are gone, it now imports `utils.binfile` and
  `params` directly (`tests/test_binfile_sync.py` removed). Ruff config pinned
  (`select = ["E", "F"]`, `target-version = "py39"`) so CI no longer depends on the ruff
  version. First release of `develop` to `main` (without the developer-only files).

**2026-09-14/15 (Baptiste)**
- Vec analysis: PSTH bin count is now `round(duration / bin_size)` — `int()` truncated
  27.999… to 27 on 40 Hz stimuli, so PSTHs randomly came out one bin short (regression test
  from the real trigger samples). New `repetitions=(start, stop)` option in
  `build_spikes_per_sequence_dict` / notebook A to keep only a range of repetitions.
- Standard preamble is now **grey + the four squares** (5 frames). The F orientation test is
  deliberately left to the user's own stimulus code (reserved key `9999`), since its purpose
  is to debug that code. Preamble keys use the same repetition-digit width as the stimulus
  they are prepended to; `utils.check_standard_preamble(bin, vec)` verifies any pair **by
  content** (which frame each square key points at, stimulus rows inside the bin) and is run
  automatically after `prepend_standard_preamble`. The four-squares check in notebook A works
  on any preamble-prefixed recording.
- Notebook 4 split into **4a Cell quality** (RPV, STA review, optional chirp review →
  `cell_quality` dict saved once per experiment, reviews saved after every answer and
  resumable) and **4b Cell typing** (loads the quality dict + saved chirp rasters, DOS split,
  clustering). Legacy `_selected_cells_for_clustering.pkl` dropped.
- Cluster summary figure: the "RF ellipses" overlay was vertically mirrored relative to the
  per-cell STAs (`contour` vs `imshow` y direction) — fixed.
- Notebook B: intro on what vec / bin are; section on natural images (dataset-wide
  normalisation to mean 0.5, clip instead of min-max; control of displayed size per rig).
- `.claude/` untracked; RF visualisation help files moved to
  `ResourcesAndTools/ReceptiveFIeldAnalysisTool`.

**2026-08-06 (Baptiste)**
- Quality pass on cell typing and cluster views; fixed `n_digit_for_rep` not passed through
  in the standard vec analysis; resources regrouped under `ResourcesAndTools/`.

**2026-07-29 (Baptiste)**
- STA figures: mask in grey with a 0 floor, yellow scale bar on the mask, symmetric colour
  bars with labels; ellipse fitted on the mask also for SWN; the `_std` vec is the default
  everywhere.

**2026-07-26 (Guilhem, Baptiste)**
- Notebook B: how to build a stimulus, the F frame. All 4 auxiliary trigger channels are read
  and used for sanity checks in the standard vec analysis.

**2026-07-13 (Awen, Baptiste) — first beta test**
- Hardcoded paths and cross-machine access fixed; repeated-sequence analysis for SWN;
  smoother handling of rigs other than MEA 2/3; spatial-mask description in notebook 2;
  physically sensible trigger threshold in preprocessing; all parameters (including
  BinFile's) moved to `params.py`.

**2026-06-26 (Baptiste) — standard vec analysis**
- Vec analysis standardised, functions moved to `utils`; DG and chirp analyses rebuilt on it
  (any number of repetitions). Typing notebook: interactive DS selection, in-place figure
  review, robust to cells without chirp response, deterministic clustering, DS cells clustered
  after non-DS with the largest IDs. `utils` became a package; `params.x` always passed
  explicitly; unit tests for the vec analysis; vec files in a dedicated folder; RF
  coordinates example in notebook 2; new preprocessing (spyking-circus part untested); SWN
  integration; backup guide and reminders; RF scale bars; reliability examples; vec columns
  drawn as stimulus traces.

**Phase II — 2026-01-26 → 2026-02-27 (Chiara, with Baptiste, Guilhem)**
- Loading helpers moved to `utils` (`prompt_user_for_recording`, `create_analysis_directory`,
  `find_analysis_directory`, `load_spike_times`, `load_stim_onset_from_triggers_path`).
- Checkerboard notebook: response extraction separated from plotting
  (`extract_all_cell_responses_to_repeated_sequences`, `plot_all_rasters`,
  `plot_raster_and_psth`).
- STA: `extract_from_sequence` / `compute_3D_sta` with explicit mandatory arguments; the many
  `analyse_sta_*` versions merged into one `rf_analysis` with a **standard method**
  (median-normalised 3D STA, `preprocess_fitting_standard` smoothing + thresholding,
  `get_sta_components` locating the RF from the std mask, ellipse drawn at 2 σ); physical
  units (`extend_sta_analysis_to_physical_units`, ellipse area / diameter, SNR, delay);
  `plot_sta_fitted_with_ellipse` with mask, markers, colour bars, scale bar, RF quantification
  text; `plot_all_stas` overview; figures saved in several formats incl. svg.
- Fitting methods compared (`ResourcesAndTools/STA_analysis_comparison.py`).

**Phase I — 2025 (Ron, Steeve)**
- Notebooks 2 and 3 refactored into thin notebooks + companion modules; `params` reorganised;
  conda `env/standard_analysis_pipeline.yml`; first unit tests (`tests/test_utils.py`) and a
  GitHub Action running them; `develop` / `main` branching model; ruff pass.
