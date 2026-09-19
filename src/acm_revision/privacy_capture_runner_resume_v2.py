from __future__ import annotations

from . import defense_dispatch
from . import fl_resume
from . import privacy_capture_runner as base
from . import privacy_capture_runner_v2 as capture_v2

# Make privacy_capture_runner use the resume-capable FL implementation.
base.fl = fl_resume

# Defense70 adds scalable implicit full/layer-update transforms. Patch the
# symbol imported by fl_resume so every formal defended update is transformed
# before aggregation while legacy experiments keep their original dispatcher.
fl_resume.apply_defense_to_state = defense_dispatch.apply_defense_to_state

# Keep the existing robust v2 capture writer:
# local spool -> fsync -> atomic NFS publish + retry.
base._save_capture_npz_factory = capture_v2._save_capture_npz_factory_v2


def main() -> None:
    base.main()


if __name__ == "__main__":
    main()
