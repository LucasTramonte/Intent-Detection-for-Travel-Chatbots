from __future__ import annotations

from eda_report import main as run_eda_report
from generate_stress_test import main as run_stress_generation
from robustness_experiments import main as run_robustness_experiments


def main() -> None:
    run_eda_report()
    run_robustness_experiments()
    run_stress_generation()
    print("Exploratory analysis finished. Outputs saved to analysis/outputs/")


if __name__ == "__main__":
    main()
