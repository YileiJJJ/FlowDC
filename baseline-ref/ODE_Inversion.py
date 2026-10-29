"""ODE-Inv reference entry point, adapted from Multi-Instruction/ODE_Inversion.py."""

from baseline_utils import run_inversion_baseline


if __name__ == "__main__":
    run_inversion_baseline("ODE-Inv", default_invert_tmax=24)
