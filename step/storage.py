"""Transactional storage; never deserialize legacy pickle files."""

import json
import os
from datetime import date, datetime
from pathlib import Path
import sqlite3
import sys
from uuid import uuid4

import numpy as np


def default_data_path():
    if sys.platform == "win32":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local"))
    elif sys.platform == "darwin":
        root = Path.home() / "Library/Application Support"
    else:
        root = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
    return root / "STEP" / "patients.sqlite3"


def validate_patient(values):
    result = {key: str(value).strip() for key, value in values.items()}
    if not result.get("name"):
        raise ValueError("Please enter a name.")
    try:
        dob = datetime.strptime(result.get("dob", ""), "%m/%d/%Y").date()
    except ValueError as exc:
        raise ValueError("Enter a valid date of birth as MM/DD/YYYY.") from exc
    today = date.today()
    if dob > today or today.year - dob.year > 125:
        raise ValueError("Please check the date of birth.")
    age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    if result.get("age") and result["age"] != str(age):
        raise ValueError(f"Age should be {age} for this date of birth, or leave age blank.")
    result["age"] = str(age)
    result["dob"] = dob.strftime("%m/%d/%Y")
    if result.get("email") and ("@" not in result["email"] or "." not in result["email"].split("@")[-1]):
        raise ValueError("Please check the email address or leave it blank.")
    if result.get("zip") and (not result["zip"].isdigit() or len(result["zip"]) != 5):
        raise ValueError("ZIP code must be five digits or blank.")
    return result


def validate_embedding(embedding):
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if vector.shape != (128,) or not np.all(np.isfinite(vector)) or np.linalg.norm(vector) < 1e-8:
        raise ValueError("Invalid SFace embedding; please capture the face again.")
    return vector


class PatientStore:
    def __init__(self, path):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(path))
        self.connection.execute("PRAGMA secure_delete=ON")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS patients (
            id TEXT PRIMARY KEY, details TEXT NOT NULL,
            embedding TEXT, model TEXT NOT NULL DEFAULT 'sface-2021dec',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )""")

    def add(self, details, embedding=None, patient_id=None):
        details = validate_patient(details)
        encoded = None if embedding is None else json.dumps(validate_embedding(embedding).tolist())
        patient_id = patient_id or uuid4().hex[:12].upper()
        with self.connection:
            self.connection.execute(
                "INSERT INTO patients (id, details, embedding) VALUES (?, ?, ?)",
                (patient_id, json.dumps(details), encoded),
            )
        return patient_id

    def get(self, patient_id):
        row = self.connection.execute("SELECT id, details FROM patients WHERE id = ?", (patient_id.strip().upper(),)).fetchone()
        return None if row is None else {"id": row[0], **json.loads(row[1])}

    def enrolled(self):
        rows = self.connection.execute("SELECT id, embedding FROM patients WHERE embedding IS NOT NULL AND model = 'sface-2021dec'")
        return [(row[0], validate_embedding(json.loads(row[1]))) for row in rows]

    def delete(self, patient_id):
        with self.connection:
            return self.connection.execute("DELETE FROM patients WHERE id = ?", (patient_id,)).rowcount > 0

    def count(self):
        return self.connection.execute("SELECT COUNT(*) FROM patients").fetchone()[0]

    def close(self):
        self.connection.close()


def seed_demo(store):
    store.add({"name": "Alex Example (fictional)", "dob": "01/15/1995", "email": "alex@example.invalid",
               "insurance": "Demo plan", "allergies": "Fictional example: pollen",
               "problems": "None (fictional)", "visit": "<1 Year"}, patient_id="DEMO001")
    store.add({"name": "Jordan Sample (fictional)", "dob": "06/20/1988", "insurance": "Uninsured"}, patient_id="DEMO002")


def format_patient(patient):
    dob = datetime.strptime(patient["dob"], "%m/%d/%Y").date()
    today = date.today()
    age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    fields = [("User ID", patient["id"]), ("Name", patient["name"]), ("Age", age),
              ("Date of Birth", patient["dob"]), ("Email", patient.get("email", "")),
              ("Phone Number", patient.get("phone", "")), ("Address", patient.get("address", "")),
              ("City / State / ZIP", " ".join(patient.get(key, "") for key in ("city", "state", "zip"))),
              ("Insurance", patient.get("insurance", "Uninsured")),
              ("Allergies", patient.get("allergies", "") or "None reported"),
              ("Medical Complications", patient.get("problems", "") or "None reported"),
              ("Most Recent Medical Visit (at enrollment)", patient.get("visit", "") or "Not specified")]
    return "\n".join(f"{label}: {value}" for label, value in fields)
