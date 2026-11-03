"""Tests for comparison CSV coordinates and automatic common-spot matching."""

import csv
import tempfile
import unittest
from pathlib import Path

from src.package.comparison import (
    _auto_match_common_spots,
    _registered_source_slots,
    _read_csv,
    _write_common_coordinate_csv,
)
from src.package.solar_coordinates import Orientation, SolarGeometry, SolarTransform


class ComparisonMatchingTests(unittest.TestCase):
    """Verify coordinate-specific exports and one-to-one cross-source matches."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_statistics(
        self,
        path: Path,
        rows: list[dict[str, str]],
    ) -> None:
        """Write the minimum statistics CSV required by the matching routine."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "sunspot_id",
            "label",
            "centroid_x_common",
            "centroid_y_common",
        ]
        with path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def test_common_csv_converts_centroid_and_bounding_box(self) -> None:
        image_path = self.root / "image.csv"
        common_path = self.root / "common.csv"
        image_path.write_text(
            "sunspot_id,label,centroid_x_px,centroid_y_px,"
            "bbox_x_px,bbox_y_px,bbox_width_px,bbox_height_px,area_px\n"
            "1,7,125,75,120,70,10,10,64\n",
            encoding="utf-8",
        )
        transform = SolarTransform(
            SolarGeometry(center=(100.0, 100.0), radius=50.0),
            Orientation((0.0, 1.0)),
        )

        _write_common_coordinate_csv(image_path, common_path, transform)

        fieldnames, rows = _read_csv(common_path)
        self.assertNotIn("centroid_x_px", fieldnames)
        self.assertNotIn("bbox_width_px", fieldnames)
        self.assertIn("centroid_x_common", fieldnames)
        self.assertIn("bbox_width_common", fieldnames)
        self.assertEqual(rows[0]["centroid_x_common"], "0.500000")
        self.assertEqual(rows[0]["centroid_y_common"], "0.500000")
        self.assertEqual(rows[0]["bbox_width_common"], "0.200000")
        self.assertEqual(rows[0]["area_px"], "64")

    def test_matching_assigns_common_id_and_keeps_unmatched_spots(self) -> None:
        source_rows = {
            "observatory_a": [
                {
                    "sunspot_id": "1",
                    "label": "3",
                    "centroid_x_common": "0.100000",
                    "centroid_y_common": "0.200000",
                },
                {
                    "sunspot_id": "2",
                    "label": "8",
                    "centroid_x_common": "-0.400000",
                    "centroid_y_common": "0.000000",
                },
            ],
            "observatory_b": [
                {
                    "sunspot_id": "1",
                    "label": "5",
                    "centroid_x_common": "0.105000",
                    "centroid_y_common": "0.202000",
                },
                {
                    "sunspot_id": "2",
                    "label": "9",
                    "centroid_x_common": "0.700000",
                    "centroid_y_common": "0.700000",
                },
            ],
        }
        sources = []
        for source_id, rows in source_rows.items():
            sources.append(
                {
                    "source_id": source_id,
                    "source_name": source_id,
                    "image": f"{source_id}.tif",
                    "output_dir": str(self.root / source_id),
                }
            )
            for coordinate_folder in ("common_coordinates", "image_coordinates"):
                self._write_statistics(
                    self.root
                    / coordinate_folder
                    / source_id
                    / "sunspot_statistics.csv",
                    rows,
                )

        summary = _auto_match_common_spots(self.root, sources)

        self.assertEqual(summary["matched_group_count"], 1)
        self.assertEqual(summary["matched_detection_count"], 2)
        self.assertEqual(summary["unmatched_detection_count"], 2)

        for coordinate_folder in ("common_coordinates", "image_coordinates"):
            first_fields, first_rows = _read_csv(
                self.root
                / coordinate_folder
                / "observatory_a"
                / "sunspot_statistics.csv"
            )
            second_fields, second_rows = _read_csv(
                self.root
                / coordinate_folder
                / "observatory_b"
                / "sunspot_statistics.csv"
            )
            self.assertIn("common_sunspot_id", first_fields)
            self.assertEqual(first_rows[0]["common_sunspot_id"], "CS0001")
            self.assertEqual(first_rows[0]["match_status"], "matched")
            self.assertEqual(first_rows[1]["match_status"], "unmatched")
            self.assertEqual(second_rows[0]["common_sunspot_id"], "CS0001")
            self.assertEqual(second_rows[1]["match_status"], "unmatched")


    def test_summary_slots_keep_source_registration_order(self) -> None:
        """Missing observatories retain their registered tile positions."""
        registered = [
            {"source_id": "first"},
            {"source_id": "second"},
            {"source_id": "third"},
            {"source_id": "fourth"},
        ]
        successful = [
            {"source_id": "fourth", "value": 4},
            {"source_id": "first", "value": 1},
        ]

        slots = _registered_source_slots(successful, registered)

        self.assertEqual(
            [None if slot is None else slot["source_id"] for slot in slots],
            ["first", None, None, "fourth"],
        )

    def test_matching_group_respects_maximum_pairwise_distance(self) -> None:
        """A chain of close neighbors cannot form a group wider than tolerance."""
        source_points = {
            "observatory_a": 0.00,
            "observatory_b": 0.02,
            "observatory_c": 0.04,
        }
        sources = []
        for source_id, x in source_points.items():
            sources.append(
                {
                    "source_id": source_id,
                    "source_name": source_id,
                    "image": f"{source_id}.tif",
                    "output_dir": str(self.root / source_id),
                }
            )
            for coordinate_folder in ("common_coordinates", "image_coordinates"):
                self._write_statistics(
                    self.root / coordinate_folder / source_id / "sunspot_statistics.csv",
                    [{
                        "sunspot_id": "1",
                        "label": "1",
                        "centroid_x_common": f"{x:.6f}",
                        "centroid_y_common": "0.000000",
                    }],
                )

        summary = _auto_match_common_spots(self.root, sources, tolerance=0.025)

        self.assertEqual(summary["matched_group_count"], 1)
        self.assertEqual(summary["matched_detection_count"], 2)
        self.assertEqual(summary["unmatched_detection_count"], 1)

if __name__ == "__main__":
    unittest.main()
