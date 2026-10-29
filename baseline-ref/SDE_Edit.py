"""SDEdit reference entry point, adapted from Multi-Instruction/SDE_Edit.py."""

from baseline_utils import run_inversion_baseline


if __name__ == "__main__":
    run_inversion_baseline("SDEdit", default_invert_tmax=21)
