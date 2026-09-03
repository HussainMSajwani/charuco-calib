import unittest

import cv2
import numpy as np

import pipeline


class ViewSelectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.width, cls.height = 1920, 1080
        cls.object_points = pipeline.make_board().getChessboardCorners()
        cls.ids = np.arange(len(cls.object_points), dtype=np.int32)
        cls.K = np.array([[1300.0, 0.0, 959.5],
                          [0.0, 1300.0, 539.5],
                          [0.0, 0.0, 1.0]])

    def record(self, number, tx, ty, depth, rx, ry, rz, sharpness=200.0):
        points, _ = cv2.projectPoints(
            self.object_points.astype(np.float64),
            np.array([rx, ry, rz], dtype=np.float64),
            np.array([tx, ty, depth], dtype=np.float64),
            self.K,
            np.zeros(5),
        )
        return (number, points.reshape(-1, 2).astype(np.float32),
                self.ids.copy(), sharpness)

    def varied_records(self):
        records = []
        for index in range(36):
            phase = 2 * np.pi * index / 36
            records.append(self.record(
                index * 20,
                0.24 * np.cos(phase),
                0.12 * np.sin(phase),
                0.75 + 0.55 * (index % 6) / 5,
                -0.55 + 1.10 * (index % 7) / 6,
                -0.45 + 0.90 * (index % 5) / 4,
                -0.8 + 1.6 * index / 35,
                150 + index,
            ))
        return np.array(records, dtype=object)

    def test_selection_spans_image_scale_roll_and_tilt(self):
        records = self.varied_records()
        selected, report = pipeline.select_views(
            records, self.width, self.height, 16, 55, min_frame_gap=12)
        self.assertEqual(len(selected), 16)
        self.assertGreater(report["centroid_x_range"], 0.35)
        self.assertGreater(report["centroid_y_range"], 0.15)
        self.assertGreater(report["log_area_range"], 0.7)
        self.assertGreater(report["roll_coverage_deg"], 60)
        self.assertGreater(report["normal_x_range"], 0.35)
        self.assertGreater(report["normal_y_range"], 0.25)

    def test_frame_gap_and_exclusion_are_hard_constraints(self):
        records = np.array([
            self.record(index, 0.002 * index, 0, 1.0, 0, 0, 0)
            for index in range(40)
        ], dtype=object)
        selected, _ = pipeline.select_views(
            records, self.width, self.height, 20, 55,
            min_frame_gap=5, excluded=[0, 20])
        frames = np.array([records[index][0] for index in selected])
        distances = np.abs(frames[:, None] - frames[None, :])
        distances += np.eye(len(frames), dtype=int) * 1000
        self.assertGreaterEqual(distances.min(), 5)
        self.assertNotIn(0, selected)
        self.assertNotIn(20, selected)


if __name__ == "__main__":
    unittest.main()
