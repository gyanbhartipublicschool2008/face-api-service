import base64
import io
from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
from PIL import Image

app = Flask(__name__)
CORS(app)


def base64_to_image(b64_str):
  if ',' in b64_str:
    b64_str = b64_str.split(',')[1]
  image_data = base64.b64decode(b64_str)
  img = Image.open(io.BytesIO(image_data)).convert('L')  # Grayscale
  return img


def is_valid_image(img):
  arr = np.array(img)
  # Standard deviation check: Diwal/plain background reject karne ke liye
  std = np.std(arr)
  if std < 18.0:
    return False
  return True


def get_image_features(img):
  # Standardized 128x128 face vector
  resized = img.resize((128, 128))
  arr = np.array(resized, dtype=np.float32)

  # Normalize lighting
  arr = (arr - np.mean(arr)) / (np.std(arr) + 1e-5)

  # 1. Pixel spatial vector
  spatial_vec = arr.flatten()

  # 2. Histogram feature vector (64 bins)
  hist, _ = np.histogram(arr, bins=64, range=(-3, 3), density=True)

  return spatial_vec, hist


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

    master_img = base64_to_image(master_b64)
    live_img = base64_to_image(live_b64)

    # Blank / Plane wall rejection
    if not is_valid_image(master_img):
      return jsonify({
          'matched': False,
          'message': 'Master photo me chehra theek se nahi dikha!',
          'similarity': 0,
      })

    if not is_valid_image(live_img):
      return jsonify({
          'matched': False,
          'message': 'Diwal ya plane background detect hua! Chehra dikhayein.',
          'similarity': 0,
      })

    # Feature extraction
    m_spatial, m_hist = get_image_features(master_img)
    l_spatial, l_hist = get_image_features(live_img)

    # Cosine Similarity for spatial texture
    spatial_sim = np.dot(m_spatial, l_spatial) / (
        np.linalg.norm(m_spatial) * np.linalg.norm(l_spatial)
    )

    # Histogram Correlation for tone/distribution
    hist_sim = np.corrcoef(m_hist, l_hist)[0, 1]

    # Weighted score (0 to 100)
    score = (max(0, spatial_sim) * 0.6) + (max(0, hist_sim) * 0.4)
    similarity = round(float(score * 100), 1)

    # 70% threshold
    is_matched = bool(similarity >= 70.0)

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
