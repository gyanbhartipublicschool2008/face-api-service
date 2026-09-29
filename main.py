import base64
import cv2
from flask import Flask, jsonify, request
from flask_cors import CORS
import mediapipe as mp
import numpy as np

app = Flask(__name__)
CORS(app)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True, max_num_faces=1, refine_landmarks=True
)


def base64_to_cv2(b64_str):
  if ',' in b64_str:
    b64_str = b64_str.split(',')[1]
  data = base64.b64decode(b64_str)
  arr = np.frombuffer(data, np.uint8)
  return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def get_face_vector(img):
  rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  results = face_mesh.process(rgb)
  if not results.multi_face_landmarks:
    return None
  landmarks = results.multi_face_landmarks[0].landmark
  # 468 landmark coordinates (x, y, z)
  vec = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
  return vec


@app.route('/compare', methods=['POST'])
def compare_faces():
  try:
    data = request.get_json()
    master_b64 = data.get('masterPhoto')
    live_b64 = data.get('livePhoto')

    if not master_b64 or not live_b64:
      return jsonify(
          {'matched': False, 'message': 'Photos missing', 'similarity': 0}
      )

    master_img = base64_to_cv2(master_b64)
    live_img = base64_to_cv2(live_b64)

    master_vec = get_face_vector(master_img)
    if master_vec is None:
      return jsonify({
          'matched': False,
          'message': 'Master photo me chehra detect nahi hua!',
          'similarity': 0,
      })

    live_vec = get_face_vector(live_img)
    if live_vec is None:
      # Diwal, pankha ya koi vastu aane par reject
      return jsonify({
          'matched': False,
          'message': (
              'Live photo me koi chehra nahi mila! (Diwal/Object rejected)'
          ),
          'similarity': 0,
      })

    # Cosine Similarity between face geometric landmarks
    cos_sim = np.dot(master_vec, live_vec) / (
        np.linalg.norm(master_vec) * np.linalg.norm(live_vec)
    )
    similarity = round(float(cos_sim) * 100, 1)

    # Threshold: Chehra match hone ke liye 85% se upar zaroori hai
    is_matched = bool(similarity >= 85.0)

    return jsonify({
        'matched': is_matched,
        'similarity': similarity,
        'message': (
            'Face Verified!'
            if is_matched
            else 'Face Mismatch! (Doosra insaan detect hua)'
        ),
    })

  except Exception as e:
    return jsonify(
        {'matched': False, 'message': 'Server Error: ' + str(e), 'similarity': 0}
    )


if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5000)
