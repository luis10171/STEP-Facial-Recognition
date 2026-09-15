"""Regression coverage for the original crash and data consistency cases."""
from datetime import date
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch, Mock

import numpy as np

from step.recognition import Camera, FaceEngine, best_match, MODEL_DIR
from step.storage import PatientStore, seed_demo, validate_patient


def vector(index=0):
    result = np.zeros(128, dtype=np.float32)
    result[index] = 1
    return result


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.store = PatientStore(":memory:")
        self.addCleanup(self.store.close)

    def test_new_database_is_empty(self):
        self.assertEqual(self.store.count(), 0)
        self.assertIsNone(self.store.get("missing"))
        self.assertEqual(self.store.enrolled(), [])

    def test_roundtrip_and_delete_keep_identity_attached(self):
        first = self.store.add({"name": "First", "dob": "01/01/2000"}, vector(0))
        second = self.store.add({"name": "Second", "dob": "01/01/2001"}, vector(1))
        self.store.delete(first)
        match, _ = best_match(vector(1), self.store.enrolled())
        self.assertEqual(match, second)
        self.assertEqual(self.store.get(match)["name"], "Second")

    def test_immediate_persistence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested/patients.sqlite3"
            writer = PatientStore(path)
            try:
                record_id = writer.add({"name": "Fictional", "dob": "02/02/2000"}, vector())
                reader = PatientStore(path)
                try:
                    self.assertEqual(reader.get(record_id)["name"], "Fictional")
                finally:
                    reader.close()
            finally:
                writer.close()

    def test_invalid_embedding_does_not_save_patient(self):
        for invalid in (np.zeros(128), np.ones(127), np.full(128, np.nan)):
            with self.assertRaises(ValueError):
                self.store.add({"name": "Example", "dob": "01/01/2000"}, invalid)
        self.assertEqual(self.store.count(), 0)

    def test_duplicate_id_does_not_overwrite(self):
        self.store.add({"name": "First", "dob": "01/01/2000"}, patient_id="SAME")
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.add({"name": "Second", "dob": "01/01/2000"}, patient_id="SAME")
        self.assertEqual(self.store.get("same")["name"], "First")

    def test_demo_has_no_biometrics(self):
        seed_demo(self.store)
        self.assertEqual(self.store.count(), 2)
        self.assertEqual(self.store.enrolled(), [])
        self.assertIn("fictional", self.store.get(" demo001 ")["name"])


class ValidationTests(unittest.TestCase):
    def test_required_fields_and_invalid_dates(self):
        for fields in ({}, {"name": "A", "dob": "02/30/2000"}, {"name": "A", "dob": "01/01/2999"}):
            with self.assertRaises(ValueError):
                validate_patient(fields)

    def test_age_is_calculated(self):
        patient = validate_patient({"name": "Example", "dob": "01/01/2000"})
        self.assertEqual(patient["age"], str(date.today().year - 2000))

    def test_inconsistent_age_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_patient({"name": "Example", "dob": "01/01/2000", "age": "1"})


class RecognitionTests(unittest.TestCase):
    def test_empty_database_and_unknown_face(self):
        self.assertEqual(best_match(vector(), []), (None, None))
        self.assertIsNone(best_match(vector(), [("OTHER", vector(1))])[0])

    def test_nearest_match_not_first_match(self):
        near = vector(0) + vector(1)
        self.assertEqual(best_match(vector(1), [("FIRST", near), ("BEST", vector(1))])[0], "BEST")

    def test_each_face_gets_its_own_identity(self):
        enrolled = [("ALPHA", vector(0)), ("BETA", vector(1))]
        self.assertEqual([best_match(vector(i), enrolled)[0] for i in (0, 1)], ["ALPHA", "BETA"])

    def test_threshold_validation(self):
        for threshold in (2, float("nan")):
            with self.assertRaises(ValueError):
                best_match(vector(), [], threshold)

    def test_failed_camera_is_released(self):
        capture = Mock()
        capture.isOpened.return_value = False
        with patch("step.recognition.cv2.VideoCapture", return_value=capture):
            camera = Camera(99)
            with self.assertRaises(RuntimeError):
                camera.read()
            capture.release.assert_called_once()
            self.assertIsNone(camera.capture)

    def test_failed_frame_is_released(self):
        capture = Mock()
        capture.isOpened.return_value = True
        capture.read.return_value = (False, None)
        with patch("step.recognition.cv2.VideoCapture", return_value=capture):
            with self.assertRaises(RuntimeError):
                Camera().read()
            capture.release.assert_called_once()

    def test_multiple_faces_rejected_before_encoding(self):
        engine = FaceEngine.__new__(FaceEngine)
        engine.detect = Mock(return_value=[object(), object()])
        with self.assertRaisesRegex(ValueError, "2 faces"):
            engine.single_face(np.zeros((480, 640, 3), np.uint8))

    @unittest.skipUnless((MODEL_DIR / "face_recognition_sface_2021dec.onnx").exists(), "Download models for integration check")
    def test_real_models_reject_blank_image(self):
        engine = FaceEngine()
        with self.assertRaisesRegex(ValueError, "0 faces"):
            engine.single_face(np.zeros((480, 640, 3), np.uint8))


if __name__ == "__main__":
    unittest.main()
