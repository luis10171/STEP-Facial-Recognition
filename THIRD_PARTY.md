# Third-party components

Dependencies are installed separately, not vendored into this repository.

| Component | Source | License |
| --- | --- | --- |
| PySimpleGUI 6.3.0.1 | https://github.com/PySimpleGUI/PySimpleGUI | LGPL-3.0, as declared by this package |
| OpenCV / opencv-python | https://github.com/opencv/opencv-python | Consult wheel notices for OpenCV, wrapper, and included libraries |
| NumPy | https://numpy.org/doc/stable/license.html | BSD-3-Clause |
| YuNet model | https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet | MIT; copied in models/face_detection_yunet_LICENSE |
| SFace model | https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface | Apache-2.0; copied in models/face_recognition_sface_LICENSE |

Model download locations are pinned to the commit in `models/manifest.json`, and the hashes are taken from that revision's Git LFS pointers. Weight files are downloaded into the ignored `models/*.onnx` paths. The copied upstream license texts apply to those models, not to the STEP application code.

`STEP logo.png` identifies the New York State Science & Technology Entry Program. Its rights remain with the relevant owner; no permission to reuse that branding is granted here. The portfolio screenshot contains the same branding and fictional demo text only.
