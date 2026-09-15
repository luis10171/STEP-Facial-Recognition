"""Original STEP screens, with explicit navigation and camera lifetime."""
from pathlib import Path
import sqlite3

import cv2
import numpy as np
import PySimpleGUI as sg

from .recognition import Camera, FaceEngine, best_match
from .storage import PatientStore, default_data_path, format_patient, seed_demo, validate_patient

ROOT = Path(__file__).resolve().parent.parent
STATES = "AK AL AR AZ CA CO CT DC DE FL GA HI IA ID IL IN KS KY LA MA MD ME MI MN MO MS MT NC ND NE NH NJ NM NV NY OH OK OR PA RI SC SD TN TX UT VA VT WA WI WV WY".split()
VISITS = ["<1 Year", "1-2 Years", "3-4 Years", "4+ Years"]
FIELDS = [("Name", "name"), ("Age (or leave blank)", "age"), ("Date of Birth (MM/DD/YYYY)", "dob"),
          ("Email", "email"), ("Phone Number", "phone"), ("Address", "address"),
          ("City", "city"), ("Zip Code", "zip")]
PAGES = ["home", "terms", "upload", "form", "allergies", "search", "result", "failed", "live"]


def centered(*elements):
    return [sg.Push(), *elements, sg.Push()]


def build_window(demo):
    sg.theme("Reddit")
    sg.set_options(font=("Bahnschrift", 12))
    heading = ("Bahnschrift", 22)
    button = ("Bahnschrift", 18)
    logo = sg.Image(str(ROOT / "STEP logo.png"), pad=((0, 0), (10, 15)))
    home = [centered(logo), centered(sg.Text("Medical Database and Facial Scanner", font=heading)),
            [sg.Text("", size=(1, 1))],
            centered(sg.Button("Upload Patient to Database", font=button, size=(27, 1), key="enroll")),
            centered(sg.Button("Live Facial Recognition", font=button, size=(27, 1), key="live-start")),
            centered(sg.Button("Access Database", font=button, size=(27, 1), key="access")),
            centered(sg.Text("", key="count", pad=(0, 14)))]
    terms = [[sg.Text("Terms and Conditions", font=heading)],
             [sg.Text("This educational demo stores the details you enter and a face template locally.\n"
                      "Use fictional medical details and only your own photo or a consenting volunteer.\n"
                      "Camera images are processed in memory and are not saved by STEP.\n"
                      "You can delete a record from its details page.\n\n"
                      "By clicking Agree, you consent to this local demonstration.")],
             [sg.Button("Disagree", key="home-terms"), sg.Button("Agree", key="agree")]]
    upload = [[sg.Text("Please take a clear photo.", font=heading)],
              [sg.Image(key="enroll-image", size=(640, 480))],
              [sg.Button("Start Camera", key="camera-enroll"), sg.Button("Capture", key="capture-enroll"),
               sg.Button("Choose Photo", key="photo-enroll"), sg.Button("Cancel", key="home-upload")]]
    form = [[sg.Text("Please enter the following information", font=heading)],
            [sg.Text("Name and date of birth are required. Use fictional medical details.")]]
    for label, key in FIELDS:
        form.append([sg.Text(label + ":", size=(27, 1)), sg.Input(key=key, size=(32, 1), background_color="white")])
    form += [[sg.Text("State:", size=(27, 1)), sg.Combo(STATES, key="state", readonly=True, size=(8, 1))],
             [sg.Radio("I am uninsured", "insurance", default=True, key="uninsured", enable_events=True),
              sg.Radio("Public insurance", "insurance", key="public", enable_events=True),
              sg.Radio("Private insurance", "insurance", key="private", enable_events=True)],
             [sg.Text("Insurance Provider:", size=(27, 1)), sg.Input(key="insurance", disabled=True, size=(32, 1))],
             [sg.Button("Submit", key="submit-form"), sg.Button("Cancel", key="home-form")]]
    allergies = [[sg.Text("Allergies and Medical Problems", font=heading)],
                 [sg.Text("Enter known allergies, separated by commas. Leave blank if none.")],
                 [sg.Multiline(key="allergy", size=(60, 3), background_color="white")],
                 [sg.Text("Enter known medical complications, separated by commas.")],
                 [sg.Multiline(key="problems", size=(60, 3), background_color="white")],
                 [sg.Text("When was the most recent medical care or treatment?")],
                 [sg.Listbox(VISITS, key="visit", size=(20, 4))],
                 [sg.Button("Confirm", key="save"), sg.Button("Cancel", key="home-allergies")]]
    search = [[sg.Text("Search for a patient in the database", font=heading)],
              [sg.Text("Use the patient's User ID:"), sg.Input(key="search-id", size=(18, 1)), sg.Button("Submit", key="find-id")],
              [sg.Image(key="search-image", size=(640, 480))],
              [sg.Button("Start Camera", key="camera-search"), sg.Button("Capture", key="capture-search"),
               sg.Button("Choose Photo", key="photo-search"), sg.Button("Cancel", key="home-search")]]
    result = [[sg.Text("", key="result-title", font=heading)],
              [sg.Text("", key="result-note", size=(78, 2))],
              [sg.Multiline(key="details", size=(76, 17), disabled=True)],
              [sg.Button("Return Home", key="home-result"), sg.Button("Delete Record", key="delete")]]
    failed = [[sg.Text("User not found, please return to home page.", font=heading)],
              [sg.Button("Return Home", key="home-failed")]]
    live = [[sg.Text("Live Facial Recognition", font=heading)],
            [sg.Text("Candidate matches only. Verify identity separately; medical details stay in the database view.")],
            [sg.Image(key="live-image", size=(640, 480))], [sg.Button("Return Home", key="home-live")]]
    layouts = [home, terms, upload, form, allergies, search, result, failed, live]
    layout = [[sg.Text("FICTIONAL DEMO - records disappear when closed" if demo else "EDUCATIONAL PROTOTYPE - not for clinical use", key="banner", text_color="#1765a0")],
              [sg.Column([[sg.pin(sg.Column(page, key=name, visible=name == "home", expand_x=True))]
                          for name, page in zip(PAGES, layouts)], scrollable=True, vertical_scroll_only=True,
                         expand_x=True, expand_y=True, size=(910, 730), key="pages")],
              [sg.Text("Ready", key="status", size=(100, 2), text_color="#1765a0")]]
    return sg.Window("STEP Medical Database" + (" - Demo" if demo else ""), layout,
                     size=(960, 820), resizable=True, finalize=True)


class Application:
    def __init__(self, demo=False, camera_index=0, data_dir=None):
        path = ":memory:" if demo else (Path(data_dir) / "patients.sqlite3" if data_dir else default_data_path())
        self.store = PatientStore(path)
        self.demo = demo
        if demo:
            seed_demo(self.store)
        self.camera = Camera(camera_index)
        self.engine = None
        self.page = "home"
        self.frame = None
        self.streaming = False
        self.pending_feature = None
        self.pending_details = None
        self.result_id = None
        self.window = build_window(demo)
        self.update_count()

    def update_count(self):
        self.window["count"].update(f"{self.store.count()} records" + (" - Try User ID DEMO001 or DEMO002" if self.demo else ""))

    def navigate(self, page):
        self.camera.close()
        self.streaming = False
        self.frame = None
        for key in PAGES:
            self.window[key].update(visible=key == page)
        self.page = page
        self.window["status"].update("Ready")
        if page == "home":
            self.pending_feature = None
            self.pending_details = None
            self.result_id = None
            for key in ("enroll-image", "search-image", "live-image"):
                self.window[key].update(data=cv2.imencode(".png", np.full((480, 640, 3), 245, np.uint8))[1].tobytes())
            self.window["details"].update("")
            self.window["search-id"].update("")
            self.update_count()
        self.window.refresh()
        self.window["pages"].contents_changed()
        self.window["pages"].Widget.canvas.yview_moveto(0)

    def get_engine(self):
        if self.engine is None:
            self.window["status"].update("Loading recognition models...")
            self.window.refresh()
            self.engine = FaceEngine()
        return self.engine

    def show_patient(self, patient_id, note="Found by User ID."):
        patient = self.store.get(patient_id)
        if patient is None:
            self.navigate("failed")
            return
        self.navigate("result")
        self.result_id = patient["id"]
        self.window["result-title"].update(f"Medical Information for {patient['name']}")
        self.window["result-note"].update(note)
        self.window["details"].update(format_patient(patient))

    def accept_image(self, frame, enrollment):
        feature = self.get_engine().single_face(frame)
        if enrollment:
            self.pending_feature = feature
            self.navigate("form")
        else:
            candidate, score = best_match(feature, self.store.enrolled())
            if candidate is None:
                self.navigate("failed")
            else:
                self.camera.close()
                self.streaming = False
                patient = self.store.get(candidate)
                answer = sg.popup_yes_no(f"Possible match: {patient['name']}\nUser ID: {candidate}\n"
                                         f"Cosine similarity: {score:.3f} (not confidence)\n\n"
                                         "Verify the person's identity separately. Open this record?", title="Confirm candidate")
                if answer == "Yes":
                    self.show_patient(candidate, f"Candidate selected by operator - cosine similarity {score:.3f}.")

    def handle_event(self, event, values):
        if isinstance(event, str) and event.startswith("home-"):
            self.navigate("home")
        elif event == "enroll":
            for _, key in FIELDS:
                self.window[key].update("")
            for key in ("insurance", "allergy", "problems"):
                self.window[key].update("")
            self.window["state"].update(value="")
            self.window["visit"].update(set_to_index=[])
            self.window["uninsured"].update(True)
            self.window["insurance"].update(disabled=True)
            self.navigate("terms")
        elif event == "agree":
            self.navigate("upload")
        elif event == "access":
            self.navigate("search")
        elif event in ("uninsured", "public", "private"):
            self.window["insurance"].update(disabled=values["uninsured"])
            if values["uninsured"]:
                self.window["insurance"].update("")
        elif event in ("camera-enroll", "camera-search"):
            self.frame = self.camera.read()
            self.streaming = True
        elif event in ("capture-enroll", "capture-search"):
            if not self.streaming or self.frame is None:
                raise ValueError("Start the camera first, or choose a photo.")
            self.accept_image(self.camera.read(), event == "capture-enroll")
        elif event in ("photo-enroll", "photo-search"):
            filename = sg.popup_get_file("Choose a photo of one consenting person", file_types=(("Images", "*.png;*.jpg;*.jpeg"),))
            if filename:
                frame = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError("This file could not be read as an image.")
                scale = min(1, 1280 / max(frame.shape[:2]))
                frame = cv2.resize(frame, (round(frame.shape[1] * scale), round(frame.shape[0] * scale)))
                self.accept_image(frame, event == "photo-enroll")
        elif event == "submit-form":
            self.pending_details = validate_patient({**{key: values[key] for _, key in FIELDS}, "state": values["state"],
                                                     "insurance": "Uninsured" if values["uninsured"] else values["insurance"]})
            if not values["uninsured"] and not values["insurance"].strip():
                raise ValueError("Enter an insurance provider or select uninsured.")
            self.navigate("allergies")
        elif event == "save":
            if self.pending_feature is None or self.pending_details is None:
                raise ValueError("Please start enrollment again.")
            details = {**self.pending_details, "allergies": values["allergy"].strip(), "problems": values["problems"].strip(),
                       "visit": next(iter(values["visit"]), "Not specified")}
            patient_id = self.store.add(details, self.pending_feature)
            self.pending_feature = None
            self.pending_details = None
            self.show_patient(patient_id, "Enrollment saved. Please note the User ID below.")
        elif event == "find-id":
            self.show_patient(values["search-id"])
        elif event == "delete" and self.result_id:
            if sg.popup_yes_no("Delete this record and its face template from STEP?", title="Delete record") == "Yes":
                self.store.delete(self.result_id)
                self.navigate("home")
        elif event == "live-start":
            self.get_engine()
            self.navigate("live")
            self.frame = self.camera.read()
            self.streaming = True

    def tick_camera(self):
        if not self.streaming:
            return
        frame = self.camera.read()
        frame = cv2.resize(frame, (640, 480))
        self.frame = frame
        display = frame.copy()
        image_key = "enroll-image" if self.page == "upload" else "search-image"
        if self.page == "live":
            image_key = "live-image"
            enrolled = self.store.enrolled()
            for face in self.get_engine().detect(frame):
                candidate, _ = best_match(self.engine.encode(frame, face), enrolled)
                patient = self.store.get(candidate) if candidate else None
                label = f"Candidate: {patient['name']}" if patient else "Unknown Person"
                x, y, width, height = [int(value) for value in face[:4]]
                cv2.rectangle(display, (x, y), (x + width, y + height), (255, 136, 0), 2)
                cv2.putText(display, label, (max(0, x), max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 136, 0), 2)
        self.window[image_key].update(data=cv2.imencode(".png", display)[1].tobytes())

    def close(self):
        self.camera.close()
        self.store.close()
        self.window.close()

    def run(self):
        try:
            while True:
                event, values = self.window.read(timeout=60)
                if event == sg.WIN_CLOSED:
                    break
                try:
                    self.handle_event(event, values)
                    self.tick_camera()
                except (ValueError, RuntimeError, OSError, sqlite3.Error, cv2.error) as exc:
                    self.camera.close()
                    self.streaming = False
                    self.frame = None
                    self.window["status"].update(str(exc))
                    sg.popup_error(str(exc), title="STEP")
        finally:
            self.close()
