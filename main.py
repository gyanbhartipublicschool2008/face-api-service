import base64
import cv2
from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)

# OpenCV Pre-trained Face Detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def base64_to_cv2(b64_str):
  if "," in b64_str:
    b64_str = b64_str.split(",")[1]
  data = base64.b64decode(b64_str)
  arr = np.frombuffer(data, np.uint8)
  return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def extract_face(img):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(
      gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80)
  )
  if len(faces) == 0:
    return None
  # Sabse bada chehra lein
  x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
  face_roi = gray[y : y + h, x : x + w]
  face_resized = cv2.resize(face_roi, (150, 150))
  # Histogram Equalization (Lighting effect normalize karne ke liye)
  face_eq = cv2.equalizeHist(face_resized)
  return face_eq


@app.route("/compare", methods=["POST"])
def compare_faces():
  try:
    data = request.get_json()
    master_b64 = data.get("masterPhoto")
    live_b64 = data.get("livePhoto")

    if not master_b64 or not live_b64:
      return jsonify(
          {"matched": False, "message": "Photos missing", "similarity": 0}
      )

    master_img = base64_to_cv2(master_b64)
    live_img = base64_to_cv2(live_b64)

    master_face = extract_face(master_img)
    if master_face is None:
      return jsonify({
          "matched": False,
          "message": "Master photo me chehra detect nahi hua!",
          "similarity": 0,
      })

    live_face = extract_face(live_img)
    if live_face is None:
      return jsonify({
          "matched": False,
          "message": (
              "Live photo me koi chehra nahi mila! (Diwal/Object rejected)"
          ),
          "similarity": 0,
      })

    # 1. 2D Histogram Correlation Compare
    hist1 = cv2.calcHist([master_face], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([live_face], [0], None, [256], [0, 256])
    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
    hist_sim = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    # 2. Template Matching Similarity
    res = cv2.matchTemplate(master_face, live_face, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)

    # Combined score
    combined_score = (max(0, hist_sim) * 0.4) + (max(0, max_val) * 0.6)
    similarity = round(float(combined_score * 100), 1)

    # Threshold: Match ke liye 70% se zyada hona zaroori hai
    is_matched = bool(similarity >= 70.0)

    return jsonify({
        "matched": is_matched,
        "similarity": similarity,
        "message": (
            "Face Verified!"
            if is_matched
            else "Face Mismatch! (Doosra insaan detect hua)"
        ),
    })

  except Exception as e:
    return jsonify(
        {"matched": False, "message": "Server Error: " + str(e), "similarity": 0}
    )


if __name__ == "__main__":
  app.run(host="0.0.0.0", port=5000)
