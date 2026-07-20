"""Guard against the two copies of binfile.py drifting apart.

``utils/binfile.py`` (analysis pipeline) and ``StimulusDisplayer/binfile.py`` (standalone
stimulus preview tool, its own git repository) must stay IDENTICAL. They were allowed to
diverge once already, which left the displayer unable to open files for a rig the pipeline
had learned about.

The file is dependency-free on purpose: rig settings can be passed via
``BinFile(..., rig_settings=...)`` and ``params`` is only imported when they are not. That
is what lets the same file serve both repositories.

If StimulusDisplayer is not checked out next to the pipeline, the test is skipped.

Run just these tests:
    python -m unittest tests.test_binfile_sync
"""

import os
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_BINFILE = os.path.join(REPO_ROOT, "utils", "binfile.py")
DISPLAYER_BINFILE = os.path.join(REPO_ROOT, "StimulusDisplayer", "binfile.py")


class TestBinFileSync(unittest.TestCase):
    def test_the_two_binfile_copies_are_identical(self):
        """StimulusDisplayer/binfile.py must be a verbatim copy of utils/binfile.py."""
        if not os.path.isfile(DISPLAYER_BINFILE):
            self.skipTest("StimulusDisplayer is not checked out next to the pipeline")

        with open(PIPELINE_BINFILE) as f:
            pipeline_source = f.read()
        with open(DISPLAYER_BINFILE) as f:
            displayer_source = f.read()

        self.assertEqual(
            pipeline_source,
            displayer_source,
            "utils/binfile.py and StimulusDisplayer/binfile.py have drifted apart.\n"
            "They must stay identical: copy utils/binfile.py over the StimulusDisplayer one\n"
            "(and commit it in the StimulusDisplayer repository too).",
        )

    def test_binfile_does_not_import_params_at_module_level(self):
        """The shared file must stay usable without the pipeline's params.py."""
        with open(PIPELINE_BINFILE) as f:
            lines = f.read().splitlines()

        # A module-level import is one that is not indented (inside a function it is fine:
        # BinFile imports params lazily, only when rig_settings is not given).
        module_level_params_imports = [
            line for line in lines if line.startswith(("import params", "from params"))
        ]
        self.assertEqual(
            module_level_params_imports,
            [],
            "binfile.py must not import params at module level: the StimulusDisplayer copy "
            "has no params.py. Pass rig settings via BinFile(..., rig_settings=...) instead.",
        )


if __name__ == "__main__":
    unittest.main()
