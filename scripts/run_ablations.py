import sys
import os
import json
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ablation.ablation_runner import AblationRunner


def main():
    print("=" * 60)
    print("RUNNING ABLATION STUDY SUITE")
    print("=" * 60)

    config_path = "./configs/default_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    runner = AblationRunner(config)
    report = runner.run_full_ablation_suite()

    print("\nAblation studies completed successfully.")
    print("Report Summary:")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
