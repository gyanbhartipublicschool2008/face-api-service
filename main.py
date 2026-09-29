import base64
import cv2
import numpy as np
import face_recognition
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def base64_to_cv2_img(b64_string):
    if ',' in b64_string:
        b64_string = b64_string.split(',')[1]
    img_data = base64.b64decode(b64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

@app.route('/compare', methods=['POST'])
def compare_faces():
    data = request.get_json()
    master_b64 = data.get('masterPhoto')
    live_b64 = data.get('livePhoto')

    if not master_b64 or not live_b64:
        return jsonify({"matched": False, "message": "Photos missing", "similarity": 0})

    try:
        master_img = base64_to_cv2_img(master_b64)
        rgb_master = cv2.cvtColor(master_img, cv2.COLOR_BGR2RGB)
        master_encodings = face_recognition.face_encodings(rgb_master)

        if len(master_encodings) == 0:
            return jsonify({"matched": False, "message": "Master photo me chehra detect nahi hua!", "similarity": 0})

        live_img = base64_to_cv2_img(live_b64)
        rgb_live = cv2.cvtColor(live_img, cv2.COLOR_BGR2RGB)
        live_encodings = face_recognition.face_encodings(rgb_live)

        if len(live_encodings) == 0:
            return jsonify({"matched": False, "message": "Live photo me koi chehra nahi mila! (Diwal/Object rejected)", "similarity": 0})

        master_face = master_encodings[0]
        live_face = live_encodings[0]

        face_dist = face_recognition.face_distance([master_face], live_face)[0]
        similarity = round(max(0, (1.0 - face_dist)) * 100, 1)

        is_matched = bool(face_dist <= 0.45)

        return jsonify({
            "matched": is_matched,
            "similarity": similarity,
            "message": "Face Verified!" if is_matched else "Face Mismatch! (Doosra insaan detect hua)"
        })

    except Exception as e:
        return jsonify({"matched": False, "message": "Server Error: " + str(e), "similarity": 0})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
