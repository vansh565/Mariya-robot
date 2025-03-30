from flask import Flask, request, jsonify
import cv2
import google.generativeai as genai
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from flask_cors import CORS

from datetime import datetime

app = Flask(__name__)
CORS(app)
genai.configure(api_key="AIzaSyD2eKhcIIS33G7uuvjA3VkI93bun_aELvQ")
timetable_data = """
[DATA START]
Monday:
  - 8:00-8:55 → TCS-465 (Through Swayam) 
  - 8:55-9:50 → TCS-402 (LT-5)
  - 10:10-11:05 → Design and Analysis of Algorithms (Lab Aryabhat First Floor)

Tuesday:
  - 8:00-8:55 → Library (LIB)
  - 10:10-11:05 → Career Skills (Verbal) (LT-5)
  - 1:55-2:50 → TCS-409 (New Audi GEHU 5th Floor)

Wednesday:
  - 8:00-8:55 → TCS-403 (LT-8)(TCS-403 IS A Microprocessor class)
  - 10:10-11:05 → Design and Analysis of Algorithms (Lab Aryabhat First Floor)
  - 1:55-2:50 → TCS-409 (New Audi GEHU 5th Floor)

Thursday:
  - 8:00-8:55 → TCS-403 (LT-2)
  - 10:10-11:05 → Career Skills (Soft Skills) (LT-5)
  - 1:00-1:55 → TCS-402 (CR-7)

Friday:
  - 8:00-8:55 → TCS-408 (LT-2)
  - 10:10-11:05 → TCS-402 (CR-6)
  - 1:55-2:50 → Project-Based Learning (TBA)

Saturday:
  - 8:00-8:55 → Microprocessors Lab (B.Tech Block Lab-1)
  - 1:55-2:50 → Competitive Programming (TOC-401) (Online)
[DATA END]
"""

location_data = """
[LOCATION START]
- "Where is LT1?" → "LT1 is in the ground floor. near boys washroom"
- "Where is LT-2?" → "LT2 is in the ground floor near girls washroom "
- "Where is LT 3?" → "LT3 is in the 1st floor."
- "Where is LT 4?" → "LT4 is in the 1st floor."
-"Where is LT 5?" → "LT5 is in the 2nd floor."
-"Where is LT 6?" → "LT5 is in the 2nd floor."
-"Where is LT 7?" → "LT5 is in the   3rd floor."
-"Where is LT 8?" → "LT8 is in the 3rd floor."
- "Where is CR-1?" → "CR-1 is on the ground

 floor."
- "Where is CR-7?" → "CR-7 is on the second floor."
- "Where is New Audi?" → "New Audi is on the 5th floor of the GEHU building."
[LOCATION END]
"""
from flask import Flask, request, jsonify
import cv2
import google.generativeai as genai
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from flask_cors import CORS

from datetime import datetime
@app.route("/", methods=["GET"])
def home():
    return "Mariya Robot AI Server is Live 🚀"

app = Flask(__name__)
CORS(app)
genai.configure(api_key="AIzaSyD2eKhcIIS33G7uuvjA3VkI93bun_aELvQ")
timetable_data = """
[DATA START]
Monday:
  - 8:00-8:55 → TCS-465 (Through Swayam) 
  - 8:55-9:50 → TCS-402 (LT-5)
  - 10:10-11:05 → Design and Analysis of Algorithms (Lab Aryabhat First Floor)

Tuesday:
  - 8:00-8:55 → Library (LIB)
  - 10:10-11:05 → Career Skills (Verbal) (LT-5)
  - 1:55-2:50 → TCS-409 (New Audi GEHU 5th Floor)

Wednesday:
  - 8:00-8:55 → TCS-403 (LT-8)(TCS-403 IS A Microprocessor class)
  - 10:10-11:05 → Design and Analysis of Algorithms (Lab Aryabhat First Floor)
  - 1:55-2:50 → TCS-409 (New Audi GEHU 5th Floor)

Thursday:
  - 8:00-8:55 → TCS-403 (LT-2)
  - 10:10-11:05 → Career Skills (Soft Skills) (LT-5)
  - 1:00-1:55 → TCS-402 (CR-7)

Friday:
  - 8:00-8:55 → TCS-408 (LT-2)
  - 10:10-11:05 → TCS-402 (CR-6)
  - 1:55-2:50 → Project-Based Learning (TBA)

Saturday:
  - 8:00-8:55 → Microprocessors Lab (B.Tech Block Lab-1)
  - 1:55-2:50 → Competitive Programming (TOC-401) (Online)
[DATA END]
"""

location_data = """
[LOCATION START]
- "Where is LT1?" → "LT1 is in the ground floor. near boys washroom"
- "Where is LT-2?" → "LT2 is in the ground floor near girls washroom "
- "Where is LT 3?" → "LT3 is in the 1st floor."
- "Where is LT 4?" → "LT4 is in the 1st floor."
-"Where is LT 5?" → "LT5 is in the 2nd floor."
-"Where is LT 6?" → "LT5 is in the 2nd floor."
-"Where is LT 7?" → "LT5 is in the   3rd floor."
-"Where is LT 8?" → "LT8 is in the 3rd floor."
- "Where is CR-1?" → "CR-1 is on the ground

 floor."
- "Where is CR-7?" → "CR-7 is on the second floor."
- "Where is New Audi?" → "New Audi is on the 5th floor of the GEHU building."
[LOCATION END]
"""

@app.route('/gemini', methods=['POST'])
def generate_response():
    try:
        # Get JSON data from request body
        data = request.get_json()
        prompt = data.get("prompt", "")
        print (prompt)

        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400
        full_prompt = f"""
           
        You are a personal assistant with the ability to answer questions from any domain. However, for questions related to timetables or locations, please refer to the following specific timetable and location details to provide the most accurate information. For any other questions, feel free to use relevant external sources such as 'Gemini' to provide the best possible answers.

        - When the user asks about the timetable or location, use the information provided in this prompt to give the best response.
          - For all other inquiries, you may use external sources or your general knowledge to answer.

        If the question is about time tables or locations, respond with the best possible answer from the provided details.
            For any other questions, answer based on your general knowledge or external tools, such as "Gemini."

          Note: If I ever ask about a timetable or location, make sure to answer with those specific details, and for anything else, use relevant tools or your knowledge base.




        Timetable:
        {timetable_data}

        Locations:
        {location_data}

        User Query: {prompt}
        """

        # Generate AI response using Gemini API
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(full_prompt)
        print (response)

        # Extract response text
        ai_response = response.text if response.text else "No response generated."

        return jsonify({"text": ai_response})

    except Exception as e:
        return jsonify({"error":str(e)}), 500




# Load trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

names = {1: "Vansh", 2: "sir/mam"}

@app.route("/recognize", methods=["POST"])
def recognize():
    try:
        # Get image from request
        data = request.json["image"]
        image_data = base64.b64decode(data.split(",")[1])  # Remove Base64 prefix
        image = Image.open(BytesIO(image_data))

        # Convert to OpenCV format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # Detect faces
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(30,30))

        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        if len(faces) == 0:
            return jsonify({"name": "No face detected"})
        for (x, y, w, h) in faces:
            id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
            confidence_text = round(100 - confidence)
            name = names.get(id, "Unknown")
  
            # Store in MongoDB
        
            return jsonify({"id": id, "name": name, "confidence": confidence_text})

        return jsonify({"name": "No face detected"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000,debug=True)
@app.route('/gemini', methods=['POST'])
def generate_response():
    try:
        # Get JSON data from request body
        data = request.get_json()
        prompt = data.get("prompt", "")
        print (prompt)

        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400
        full_prompt = f"""
           
        You are a personal assistant with the ability to answer questions from any domain. However, for questions related to timetables or locations, please refer to the following specific timetable and location details to provide the most accurate information. For any other questions, feel free to use relevant external sources such as 'Gemini' to provide the best possible answers.

        - When the user asks about the timetable or location, use the information provided in this prompt to give the best response.
          - For all other inquiries, you may use external sources or your general knowledge to answer.

        If the question is about time tables or locations, respond with the best possible answer from the provided details.
            For any other questions, answer based on your general knowledge or external tools, such as "Gemini."

          Note: If I ever ask about a timetable or location, make sure to answer with those specific details, and for anything else, use relevant tools or your knowledge base.




        Timetable:
        {timetable_data}

        Locations:
        {location_data}

        User Query: {prompt}
        """

        # Generate AI response using Gemini API
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(full_prompt)
        print (response)

        # Extract response text
        ai_response = response.text if response.text else "No response generated."

        return jsonify({"text": ai_response})

    except Exception as e:
        return jsonify({"error":str(e)}), 500




# Load trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

names = {1: "Vansh", 2: "sir/mam"}

@app.route("/recognize", methods=["POST"])
def recognize():
    try:
        # Get image from request
        data = request.json["image"]
        image_data = base64.b64decode(data.split(",")[1])  # Remove Base64 prefix
        image = Image.open(BytesIO(image_data))

        # Convert to OpenCV format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # Detect faces
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(30,30))

        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        if len(faces) == 0:
            return jsonify({"name": "No face detected"})
        for (x, y, w, h) in faces:
            id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
            confidence_text = round(100 - confidence)
            name = names.get(id, "Unknown")
  
            # Store in MongoDB
        
            return jsonify({"id": id, "name": name, "confidence": confidence_text})

        return jsonify({"name": "No face detected"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000,debug=True)
