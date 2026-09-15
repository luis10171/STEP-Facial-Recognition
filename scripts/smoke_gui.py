"""Exercise the actual GUI and event handlers without a camera or personal data."""
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from step.app import Application


def main():
    app = Application(demo=True)
    try:
        def event(name):
            _, values = app.window.read(timeout=1)
            app.handle_event(name, values)
            app.window.refresh()

        event("access")
        app.window["search-id"].update("demo001")
        event("find-id")
        assert app.page == "result" and app.result_id == "DEMO001"
        event("home-result")
        event("access")
        app.window["search-id"].update("missing")
        event("find-id")
        assert app.page == "failed"
        event("home-failed")
        event("enroll")
        event("agree")
        assert app.page == "upload"
        # Simulate an already-validated face to test UI/storage, not model accuracy.
        app.pending_feature = np.ones(128, np.float32)
        app.navigate("form")
        app.window["name"].update("GUI Test (fictional)")
        app.window["dob"].update("01/01/2000")
        event("submit-form")
        assert app.page == "allergies"
        app.window["allergy"].update("Fictional allergy")
        event("save")
        assert app.page == "result" and app.store.count() == 3
        assert app.store.get(app.result_id)["allergies"] == "Fictional allergy"
        with patch("step.app.sg.popup_yes_no", return_value="Yes"):
            event("delete")
        assert app.store.count() == 2 and app.page == "home"
        event("enroll")
        event("agree")
        event("home-upload")
        assert app.pending_feature is None and app.camera.capture is None
        assert app.store.count() == 2
        print("GUI smoke passed: lookup, missing ID, enrollment, save, delete, cancel, camera cleanup.")
    finally:
        app.close()


if __name__ == "__main__":
    main()
