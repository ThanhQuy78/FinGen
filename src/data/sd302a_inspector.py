"""
Inspector and parser for the NIST SD302a ("N2N Challenge") fingerprint dataset.

Real on-disk layout (as distributed in `archive/`):

    archive/images/challengers/{DEVICE}/{IMPRESSION}/png/{SUBJECT}_{DEVICE}_{IMPRESSION}_{FRGP}.png
    e.g. archive/images/challengers/A/roll/png/00002303_A_roll_01.png

  * SUBJECT     — 8-digit zero-padded challenger id
  * DEVICE      — capture device letter A..H (all contact-based, see NIST_SD302A_SENSOR_TECH)
  * IMPRESSION  — 'roll' (rolled) / 'plain' / 'slap'
  * FRGP        — ANSI/NIST-ITL Friction Ridge Generalized Position code, 01..10

This module intentionally does NOT assume a flat directory: it walks the nested
`{DEVICE}/{IMPRESSION}/png/` tree.
"""

import os
import re
import json
from typing import Dict, List, Optional, Tuple

# NIST SD302a Sensor Technology Mapping Specification
# Sensors A-H are contact-based devices (Optical, Capacitive, Thermal-Sweep)
# Contactless/camera sensors are explicitly excluded or flagged.
NIST_SD302A_SENSOR_TECH = {
    "A": {"type": "optical", "name": "Crossmatch L Scan Guardian", "contact": True},
    "B": {"type": "optical", "name": "Identix TP-600", "contact": True},
    "C": {"type": "capacitive", "name": "AuthenTec AES3500", "contact": True},
    "D": {"type": "thermal_sweep", "name": "Atmel FingerChip", "contact": True},
    "E": {"type": "optical", "name": "SecuGen Hamster IV", "contact": True},
    "F": {"type": "capacitive", "name": "UPEK TouchStrip", "contact": True},
    "G": {"type": "optical_rolled", "name": "Identix TouchPrint 5300", "contact": True},
    "H": {"type": "capacitive_sweep", "name": "Validity VFS301", "contact": True},
}

# Sensor letter -> contiguous class index used by the MM-DiT sensor embedding.
SENSOR_TO_INDEX = {s: i for i, s in enumerate(sorted(NIST_SD302A_SENSOR_TECH))}

# ANSI/NIST-ITL Friction Ridge Generalized Position codes present in SD302a.
FRGP_TO_FINGER = {
    1: "right_thumb",   2: "right_index",  3: "right_middle",
    4: "right_ring",    5: "right_little", 6: "left_thumb",
    7: "left_index",    8: "left_middle",  9: "left_ring",
    10: "left_little",
}

# {SUBJECT}_{DEVICE}_{IMPRESSION}_{FRGP}.ext
_FILENAME_RE = re.compile(
    r"^(?P<subject>\d+)_(?P<device>[A-Ha-h])_(?P<impression>[A-Za-z]+)_(?P<frgp>\d{1,2})$"
)

IMAGE_EXTENSIONS = (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp", ".wsq")


class SD302aInspector:
    """
    Inspector and parser for NIST SD302a.
    Verifies sensor technologies, subject/finger pairing across sensors, and
    filters non-contact sensors.
    """

    def __init__(self, dataset_root: str, impressions: Optional[List[str]] = None):
        """
        Args:
            dataset_root: path to `.../images/challengers` (the directory holding A..H),
                          or any ancestor of it — the walk is recursive either way.
            impressions:  restrict to these impression types (default: all found).
        """
        self.root = dataset_root
        self.sensor_specs = NIST_SD302A_SENSOR_TECH
        self.impressions = set(i.lower() for i in impressions) if impressions else None

    def get_contact_sensors(self) -> List[str]:
        """Returns list of sensor keys that are contact-based."""
        return [k for k, v in self.sensor_specs.items() if v.get("contact", True)]

    def parse_filename(self, filename: str) -> Tuple[str, str, str, str]:
        """
        Parses an SD302a filename.

        '00002303_A_roll_01.png' -> ('00002303', '01', 'A', 'roll')

        Returns:
            (subject_id, frgp, sensor, impression); all empty strings on no match.
        """
        stem = os.path.splitext(os.path.basename(filename))[0]
        m = _FILENAME_RE.match(stem)
        if not m:
            return "", "", "", ""
        return (
            m.group("subject"),
            f"{int(m.group('frgp')):02d}",
            m.group("device").upper(),
            m.group("impression").lower(),
        )

    @staticmethod
    def frgp_to_finger_name(frgp: str) -> str:
        """Maps an FRGP code ('01'..'10') to a human-readable finger name."""
        try:
            return FRGP_TO_FINGER.get(int(frgp), f"frgp_{frgp}")
        except (TypeError, ValueError):
            return f"frgp_{frgp}"

    def scan_files(self) -> List[Dict]:
        """
        Recursively walks `self.root` and returns one record per parsable image file:
            {path, subject, frgp, sensor, impression, finger_key}
        `finger_key` = '{subject}_{frgp}' uniquely identifies a physical finger.
        """
        records: List[Dict] = []
        if not os.path.isdir(self.root):
            return records

        contact = set(self.get_contact_sensors())
        for dir_path, _, files in os.walk(self.root):
            for fname in files:
                if not fname.lower().endswith(IMAGE_EXTENSIONS):
                    continue
                subject, frgp, sensor, impression = self.parse_filename(fname)
                if not subject or sensor not in contact:
                    continue
                if self.impressions is not None and impression not in self.impressions:
                    continue
                records.append({
                    "path": os.path.join(dir_path, fname),
                    "subject": subject,
                    "frgp": frgp,
                    "sensor": sensor,
                    "impression": impression,
                    "finger_key": f"{subject}_{frgp}",
                })
        records.sort(key=lambda r: (r["subject"], r["frgp"], r["sensor"], r["impression"]))
        return records

    def build_finger_index(self) -> Dict[str, Dict[str, str]]:
        """
        Returns {finger_key: {sensor_letter: image_path}}.
        If a finger has several impressions on the same sensor, the first
        (sorted) one wins so that each (finger, sensor) cell is unambiguous.
        """
        index: Dict[str, Dict[str, str]] = {}
        for rec in self.scan_files():
            index.setdefault(rec["finger_key"], {}).setdefault(rec["sensor"], rec["path"])
        return index

    def inspect_dataset(self) -> Dict:
        """
        Scans the dataset directory and computes the subject/finger pairing matrix
        across sensors A-H.
        """
        report = {
            "root": self.root,
            "sensor_tech": self.sensor_specs,
            "total_files": 0,
            "valid_contact_files": 0,
            "subjects": [],
            "finger_pairs": {},   # '{subject}_{frgp}' -> [sensors]
            "cross_sensor_matrix": {s1: {s2: 0 for s2 in self.sensor_specs}
                                    for s1 in self.sensor_specs},
        }

        if not os.path.isdir(self.root):
            report["status"] = "Directory not found (returns structural metadata spec)"
            return report

        for dir_path, _, files in os.walk(self.root):
            report["total_files"] += sum(
                1 for f in files if f.lower().endswith(IMAGE_EXTENSIONS)
            )

        records = self.scan_files()
        report["valid_contact_files"] = len(records)

        subjects = set()
        for rec in records:
            subjects.add(rec["subject"])
            report["finger_pairs"].setdefault(rec["finger_key"], []).append(rec["sensor"])

        for sensors in report["finger_pairs"].values():
            uniq = sorted(set(sensors))
            for s1 in uniq:
                for s2 in uniq:
                    report["cross_sensor_matrix"][s1][s2] += 1

        report["subjects"] = sorted(subjects)
        report["num_subjects"] = len(subjects)
        report["num_unique_fingers"] = len(report["finger_pairs"])
        report["status"] = "ok"
        return report


if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "./archive/images/challengers"
    inspector = SD302aInspector(root)
    rep = inspector.inspect_dataset()
    print(json.dumps({
        "root": rep["root"],
        "status": rep.get("status"),
        "total_files": rep["total_files"],
        "valid_contact_files": rep["valid_contact_files"],
        "num_subjects": rep.get("num_subjects"),
        "num_unique_fingers": rep.get("num_unique_fingers"),
        "cross_sensor_matrix": rep["cross_sensor_matrix"],
    }, indent=2))
