import cv2
import face_recognition
import json
import os
import numpy as np 
from datetime import datetime 
import streamlit as st

st.title("Face Recognition Web App")
st.subheader("Detection")

def mark_user(name, user_id):
    now = datetime.now()
    time_str = now.strftime('%I:%M:%S %p')
    date_str = now.strftime('%d-%B-%Y')

    with open('time.csv', 'a') as f:
        f.write(f'\n{name}, {user_id}, {time_str}, {date_str}')

    success_msg = f"User Recognised: {name}"
    st.success(success_msg)

def recognize_user(user_name, user_id):
    with open("known_faces.json", "r") as json_file:
        known_faces = json.load(json_file)

    known_face_encodings = list(known_faces.values())
    known_face_names = list(known_faces.keys())

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        found = False

        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame, model='hog')

        for face_location in face_locations:
            face_encoding = face_recognition.face_encodings(rgb_frame, [face_location])[0]

            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            for j, is_match in enumerate(matches):
                if is_match:
                    name = known_face_names[j]
                    n, _ = name.split(",")
                    mark_user(n, user_id)
                    found = True
                    break

            if found:
                break

        if found:
            break

    cap.release()
    cv2.destroyAllWindows()

user_name = st.text_input("Full Name")
user_id = st.text_input("Student Id")
submit_btn = st.button("Submit")
att_btn = st.button("Recognise")

if submit_btn and user_name and user_id:
    mark_user(user_name, user_id)

if att_btn:
    recognize_user(user_name, user_id)