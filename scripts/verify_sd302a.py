import sys
import os
import json

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.sd302a_inspector import SD302aInspector


def main():
    print("=" * 60)
    print("NIST SD302a DATASET & SENSOR TECHNOLOGY VERIFICATION")
    print("=" * 60)

    import yaml

    config_path = sys.argv[2] if len(sys.argv) > 2 else "./configs/default_config.yaml"
    with open(config_path, "r") as f:
        ds_cfg = yaml.safe_load(f).get("dataset", {})

    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else ds_cfg.get("sd302a_root", "")
    print(f"Dataset root: {dataset_dir}")
    inspector = SD302aInspector(dataset_dir, impressions=ds_cfg.get("sd302a_impressions"))
    contact_sensors = inspector.get_contact_sensors()
    print(f"[1] Verified Contact-Based Sensors (A-H): {contact_sensors}")
    
    report = inspector.inspect_dataset()
    print("[2] Sensor Specs Mapping:")
    for sensor, spec in report["sensor_tech"].items():
        print(f"  Sensor {sensor}: {spec['name']} ({spec['type']}) -> Contact: {spec['contact']}")

    print("\n[3] Inspection Report:")
    print(f"  Total Files Found: {report['total_files']}")
    print(f"  Valid Contact Files: {report['valid_contact_files']}")
    print(f"  Unique Subjects: {report.get('num_subjects', 0)}")
    print(f"  Unique Finger Pairs: {report.get('num_unique_fingers', 0)}")
    
    output_path = "./outputs/sd302a_verification_report.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport written to: {output_path}")


if __name__ == "__main__":
    main()
