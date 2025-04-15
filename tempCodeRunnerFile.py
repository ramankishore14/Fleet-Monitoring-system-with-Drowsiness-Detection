from flask import Flask, render_template, Response, request, jsonify
import dlib
import cv2
import numpy as np
from scipy.spatial import distance as dist
from threading import Thread, Timer
import playsound
import queue
import mysql.connector

# Initialize Flask app
app = Flask(__name__)

# Define constants and variables
FACE_DOWNSAMPLE_RATIO = 1.5
RESIZE_HEIGHT = 460
thresh = 0.27
modelPath = r"C:\Users\91786\Downloads\Computer-Vision-Project-Driver-\Fleet Drowsiness Detection\models\shape_predictor_70_face_landmarks.dat"
sound_path = r"C:\Users\91786\Downloads\Computer-Vision-Project-Driver-\Fleet Drowsiness Detection\alarm.wav"

# Load dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(modelPath)

# Eye landmark indices
leftEyeIndex = [36, 37, 38, 39, 40, 41]
rightEyeIndex = [42, 43, 44, 45, 46, 47]

# Initialize global variables for drowsiness detection
blinkCount = 0
drowsy = 0
state = 0
blinkTime = 0.15  # 150ms
drowsyTime = 1.5  # 1.5 seconds
ALARM_ON = False
GAMMA = 1.5
threadStatusQ = queue.Queue()

# Configure your database connection
db_config = {
    'user': 'root',
    'password': '',
    'host': 'localhost',
    'database': 'fleet'
}

# Function to perform histogram equalization
def histogram_equalization(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray)

# Function to get landmarks from the frame
def getLandmarks(frame):
    imSmall = cv2.resize(frame, None, fx=1.0 / FACE_DOWNSAMPLE_RATIO, fy=1.0 / FACE_DOWNSAMPLE_RATIO, interpolation=cv2.INTER_LINEAR)
    rects = detector(imSmall, 0)
    if len(rects) == 0:
        return []
    
    newRect = dlib.rectangle(int(rects[0].left() * FACE_DOWNSAMPLE_RATIO),
                             int(rects[0].top() * FACE_DOWNSAMPLE_RATIO),
                             int(rects[0].right() * FACE_DOWNSAMPLE_RATIO),
                             int(rects[0].bottom() * FACE_DOWNSAMPLE_RATIO))
    
    return [(p.x, p.y) for p in predictor(frame, newRect).parts()]

# Function to check eye status
def checkEyeStatus(landmarks):
    leftEye = [landmarks[i] for i in leftEyeIndex]
    rightEye = [landmarks[i] for i in rightEyeIndex]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0

    return ear < thresh

# Calculate Eye Aspect Ratio
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to check blink status
def checkBlinkStatus(eyeStatus):
    global blinkCount, drowsy, state

    if eyeStatus:
        state += 1
    else:
        if state >= 2:
            blinkCount += 1
        state = 0

    if state >= (drowsyTime / blinkTime):
        drowsy = 1
    else:
        drowsy = 0

# Function to sound the alert
def soundAlert(sound_path, queue):
    playsound.playsound(sound_path)
    queue.get()  # Wait for a signal to stop

# Database update function
def update_alert_message(order_id, alert_message):
    try:
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()

        update_query = "UPDATE work_order SET alert_message = %s WHERE order_id = %s"
        cursor.execute(update_query, (alert_message, order_id))

        connection.commit()
        return cursor.rowcount  # Returns the number of affected rows

    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return 0

    finally:
        if connection:
            cursor.close()
            connection.close()

# Function to reset driver status
def reset_driver_status(order_id, delay):
    def reset_task():
        update_alert_message(order_id, "DRIVER NORMAL")  # Reset alert message to "DRIVER NORMAL"

    Timer(delay, reset_task).start()

def gen_frames():
    global ALARM_ON, blinkCount, drowsy, state
    last_drowsy_state = False  # Tracks the previous state of drowsiness
    capture = cv2.VideoCapture(0)

    while True:
        success, frame = capture.read()
        if not success:
            break

        height, width = frame.shape[:2]
        IMAGE_RESIZE = np.float32(height) / RESIZE_HEIGHT
        frame = cv2.resize(frame, None, fx=1 / IMAGE_RESIZE, fy=1 / IMAGE_RESIZE, interpolation=cv2.INTER_LINEAR)

        adjusted = histogram_equalization(frame)
        landmarks = getLandmarks(adjusted)

        if not landmarks:
            cv2.putText(frame, "Unable to detect face, Please check proper lighting", (10, 30),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        else:
            eyeStatus = checkEyeStatus(landmarks)
            checkBlinkStatus(eyeStatus)

            if drowsy:
                cv2.putText(frame, "! ! ! DROWSINESS ALERT ! ! !", (70, 50),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                if not last_drowsy_state:  # Play sound only when drowsiness is detected for the first time
                    last_drowsy_state = True
                    order_id = 1  # Replace with dynamic order_id
                    alert_message = "Drowsiness detected"
                    update_alert_message(order_id, alert_message)

                    # Play the sound alert
                    if not ALARM_ON:
                        ALARM_ON = True
                        thread = Thread(target=soundAlert, args=(sound_path, threadStatusQ))
                        thread.start()

                    # Reset driver status after 10 seconds
                    reset_driver_status(order_id, delay=10)
            else:
                last_drowsy_state = False  # Reset state when no drowsiness is detected
                ALARM_ON = False  # Turn off alarm when drowsiness is not detected

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/update_alert', methods=['POST'])
def update_alert():
    data = request.get_json()
    order_id = data.get('orderId')
    alert_message = data.get('alertMessage')

    if order_id and alert_message:
        rows_updated = update_alert_message(order_id, alert_message)
        if rows_updated:
            return jsonify({"success": True, "message": "Alert message updated"})
        else:
            return jsonify({"success": False, "message": "Failed to update alert message"})
    else:
        return jsonify({"success": False, "message": "Invalid data received"})

if __name__ == "__main__":
    app.run(debug=True)
