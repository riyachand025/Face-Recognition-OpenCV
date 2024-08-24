import cv2
import face_recognition
import json
import os
import numpy as np 
from datetime import datetime 
import streamlit as st
import csv

st.title("Face Recognition Web App")
st.subheader("New Face")

user_name = st.text_input("Full Name", placeholder="Enter Name...")
user_id = st.text_input("Student Id", placeholder="Enter Student Id...")

submit_btn = st.button('Submit')
run = st.button('Log Face')

FRAME_WINDOW = st.image([])

def log_face(user_name, user_id):
    known_faces = {}
    known_face_encodings = []
    known_face_names = []
    name = ""

    try:
        with open("known_faces.json", "r") as json_file:
            known_faces = json.load(json_file)
            known_face_names = list(known_faces.keys())
            known_face_encodings = [np.array(encoding) for encoding in known_faces.values()]
    except FileNotFoundError:
        known_faces = {}

    cap = cv2.VideoCapture(0)  

    while True:
        ret, frame = cap.read()
        
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame, model='hog')

        if face_locations:
            face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]

            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                matched_idx = matches.index(True)
                name = known_face_names[matched_idx]

            for top, right, bottom, left in face_locations:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                y = top - 15 if top - 15 > 15 else top + 15
                cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        FRAME_WINDOW.image(frame)

        if face_locations and name == "Unknown":
            try:
                name = f"{user_name},{user_id}"
                known_faces[name] = face_encoding.tolist()  
                known_face_names.append(name)
                known_face_encodings.append(face_encoding)

                try:
                    with open("known_faces.json", "w") as json_file:
                        json.dump(known_faces, json_file)
                        success_msg = f"Face Logged, {user_name}"
                        st.success(success_msg)
                except Exception as e:
                    st.error(f"Error writing to JSON file: {str(e)}")

                cap.release()
                cv2.destroyAllWindows()
                return
            except Exception as e:
                st.error(f"Error while saving face: {str(e)}")

def log_user(user_name, user_id):
    new_data = f"{user_name}, {user_id}\n"

    if not os.path.isfile('Users.csv'):
        with open('Users.csv', 'w') as f:
            f.write("Name, ID\n")
            f.write(new_data)
    else:
        with open('Users.csv', 'a') as f:
            f.write(new_data)

if submit_btn:
    st.success("You are now logged in!")
    log_user(user_name, user_id)

if run:
    with st.container():
        st.write("Webcam Live Feed")
        log_face(user_name, user_id)