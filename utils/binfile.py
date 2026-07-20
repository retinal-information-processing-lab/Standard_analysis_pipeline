"""Read/write the stimulus ".bin" files (the raw frames shown on the DMD).

Everything rig-dependent (DMD geometry, display polarity, optical correction) is defined
ONCE in params.py (``params.rig_params``) and read from there — nothing is hard-coded here.
Rigs that have not been implemented/tested warn and fall back to neutral defaults (see
``params.get_display_rig_params``).

This module is deliberately kept free of any import-time dependency on ``params``: the
settings can also be passed explicitly via ``BinFile(..., rig_settings={...})``, and
``params`` is only imported when they are not. That way the exact same file can be used by
the standalone StimulusDisplayer tool, which has no experiment configuration to load.
Keep the two copies identical (tests/test_binfile_sync.py checks it).
"""

from typing import Optional

import numpy as np
import os


def apply_optical_transform(frame: np.ndarray, transform: Optional[str]) -> np.ndarray:
    """Correct a frame READ from a .bin for the optical path of the rig.

    ``transform`` is the rig's "optical_transform" in params.rig_params.
    """
    if transform is None:
        return frame
    if transform == "rot90_flipud":
        return np.flipud(np.rot90(frame))
    if transform == "fliplr":
        return np.fliplr(frame)
    raise ValueError(
        f"Unknown optical_transform {transform!r} in params.rig_params. "
        "Known values: 'rot90_flipud', 'fliplr', None."
    )


def undo_optical_transform(frame: np.ndarray, transform: Optional[str]) -> np.ndarray:
    """Pre-compensate a frame about to be WRITTEN to a .bin.

    This is the inverse of apply_optical_transform, so that reading back a frame you
    wrote gives you the frame you started from.
    """
    if transform is None:
        return frame
    if transform == "rot90_flipud":
        return np.rot90(np.flipud(frame), k=3)
    if transform == "fliplr":
        return np.fliplr(frame)
    raise ValueError(
        f"Unknown optical_transform {transform!r} in params.rig_params. "
        "Known values: 'rot90_flipud', 'fliplr', None."
    )


class BinFile:
    @classmethod
    def read_header(cls, path):
        header = {}
        with open(path, mode="rb") as input_file:
            # Read image xsize.
            image_xsize_bytes = input_file.read(2)
            header["xsize"] = int.from_bytes(image_xsize_bytes, byteorder="little")
            # Read image ysize.
            image_ysize_bytes = input_file.read(2)
            header["ysize"] = int.from_bytes(image_ysize_bytes, byteorder="little")
            # Read number of images.
            nb_images_bytes = input_file.read(2)
            header["nb_images"] = int.from_bytes(nb_images_bytes, byteorder="little")
            # Read number of bits.
            nb_bits_bytes = input_file.read(2)
            header["nb_bits"] = int.from_bytes(nb_bits_bytes, byteorder="little")

        return header

    @classmethod
    def read_nb_images(cls, path):
        header = cls.read_header(path)

        return header["nb_images"]

    def __init__(
        self,
        path,
        frame_xsize,
        frame_ysize,
        rig_id,
        nb_images=0,
        reverse=False,
        mode="r",
        rig_settings=None,
    ):
        """Open a stimulus .bin for reading or writing.

        Args:
            rig_settings: dict with this rig's display settings, i.e. "max_frame_size"
                ([x, y] or None for no size check), "invert_polarity" (bool) and
                "optical_transform" ("rot90_flipud", "fliplr" or None). Leave it None in
                the analysis pipeline: the settings are then read from params.rig_params
                for ``rig_id``. Pass it explicitly to use this class without params (e.g.
                the standalone StimulusDisplayer).
        """
        self._path = path
        self._reverse = reverse
        self._mode = mode
        self._rig_id = rig_id

        # Every rig-dependent setting comes from params.py unless given explicitly. Rigs
        # that are not implemented/tested warn there (work in progress) and fall back to
        # neutral defaults. params is imported lazily so this module works without it.
        if rig_settings is None:
            import params

            rig_settings = params.get_display_rig_params(rig_id)
        rig = rig_settings
        max_frame_size = rig["max_frame_size"]
        # A rig with no known max frame size (work-in-progress) simply gets no size check.
        self._max_dimension_x, self._max_dimension_y = max_frame_size or (None, None)
        self._invert_polarity = rig["invert_polarity"]
        self._optical_transform = rig["optical_transform"]

        if self._mode == "r":
            header = self.read_header(self._path)
            self._nb_images = header["nb_images"]
            self._check_frame_size(header["xsize"], header["ysize"])
            self._frame_xsize = header["xsize"]
            self._frame_ysize = header["ysize"]
            self._nb_bits = header["nb_bits"]
            self._file = open(self._path, mode="rb")
            self._frame_nb = self._nb_images - 1
        elif self._mode == "w":
            self._nb_images = nb_images
            self._check_frame_size(frame_xsize, frame_ysize)
            self._frame_xsize = frame_xsize
            self._frame_ysize = frame_ysize
            self._nb_bits = 8
            # self._file = open(self._path, mode='w+b')
            self._file = open(self._path, mode="wb")
            self._write_header()
            self._frame_nb = -1
        else:
            raise ValueError("unknown mode value: {}".format(self._mode))

        self._counter = 0

    def _check_frame_size(self, xsize, ysize):
        """Check a frame fits this rig's DMD. Skipped when the rig's max size is unknown."""
        if self._max_dimension_x is not None:
            assert xsize <= self._max_dimension_x, (
                f"image is too big on x axis for RIG {self._rig_id} ({xsize} > {self._max_dimension_x})."
            )
        if self._max_dimension_y is not None:
            assert ysize <= self._max_dimension_y, (
                f"image is too big on y axis for RIG {self._rig_id} ({ysize} > {self._max_dimension_y})."
            )

    def __len__(self):
        return self._nb_images

    def __iter__(self):
        self._counter = 0  # i.e. reinitialization

        return self

    def __next__(self):
        if self._counter < len(self):
            frame = self.read_frame(self._counter)
            self._counter += 1
        else:
            raise StopIteration

        return frame

    @property
    def _frame_shape(self):
        return self._frame_xsize, self._frame_ysize

    @property
    def ysize(self):
        return self._frame_ysize

    @property
    def xsize(self):
        return self._frame_xsize

    @property
    def nb_frames(self):
        return self._nb_images

    @property
    def nb_bits(self):
        return self._nb_bits

    def is_readable(self):
        return self._mode == "r"

    def is_writeable(self):
        return self._mode == "w"

    def get_frame_nb(self):
        """Get the number of the latest frame appended."""
        return self._frame_nb

    def get_frame_nbs(self):
        """Get the number of frames appended."""
        return np.arange(0, len(self))

    def read_frame_as_bytes(self, frame_nb):
        """Read frame as bytes."""
        assert self.is_readable(), "not readable"
        assert 0 <= frame_nb < len(self), frame_nb
        assert self._nb_bits == 8, self._nb_bits

        # Set file's current position.
        header_byte_size = 2 * 4
        frame_byte_size = self._frame_ysize * self._frame_xsize
        byte_offset = header_byte_size + frame_byte_size * frame_nb
        self._file.seek(byte_offset)
        # Read data from file.
        frame_bytes = self._file.read(frame_byte_size)

        return frame_bytes

    def read_frame(self, frame_nb, verbose=0):
        """Read, convert to float and reshape frame."""
        assert self.is_readable(), "not readable"
        assert 0 <= frame_nb < len(self), frame_nb
        assert self._nb_bits == 8, self._nb_bits

        # Set file's current position.
        header_byte_size = 2 * 4
        frame_byte_size = self._frame_ysize * self._frame_xsize
        byte_offset = header_byte_size + frame_byte_size * frame_nb
        self._file.seek(byte_offset)
        # Read data from file.
        frame_bytes = self._file.read(frame_byte_size)
        frame_data = np.frombuffer(frame_bytes, dtype=np.uint8)
        if verbose > 1:
            print("max value = ", np.max(frame_data))

        # Convert data to float.
        frame_data = frame_data.astype(float)
        dinfo = np.iinfo(np.uint8)
        frame_data = frame_data / float(dinfo.max - dinfo.min + 1)

        # Reshape data.
        shape = (self._frame_xsize, self._frame_ysize)
        frame_data = np.reshape(frame_data, shape)

        # Undo the display polarity of this rig (frames are stored inverted on such rigs)
        if self._invert_polarity:
            frame_data = 1 - frame_data

        # Compensate the optical transformation of this rig's setup
        frame_data = apply_optical_transform(frame_data, self._optical_transform)

        return frame_data

    def _write_header(self):
        header_list = [
            self._frame_xsize,
            self._frame_ysize,
            self._nb_images,
            self._nb_bits,
        ]
        print("header_list: {}".format(header_list))
        header_array = np.array(header_list, dtype=np.int16)
        header_bytes = header_array.tobytes()
        self._file.write(header_bytes)

        return

    def append(self, frame):
        if isinstance(frame, bytes):
            assert len(frame) == self._frame_ysize * self._frame_xsize, len(frame)

            self._file.write(frame)

        else:
            # reverse polarity if necessary (to compensate polarity reversal on display)
            if self._invert_polarity:
                frame = 1 - frame

            # transform to compensate optical transformation on the setup
            frame = undo_optical_transform(frame, self._optical_transform)

            # Scale data between 0 and 255
            frame = frame * 255
            frame = frame.astype("uint8")

            frame_bytes = frame.tobytes()
            self._file.write(frame_bytes)

        self._frame_nb += 1

        return

    def flush(self):
        os.fsync(self._file.fileno())  # force write

        return

    def close(self):
        self.flush()
        self._file.close()

        return
