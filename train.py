import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import os
import csv
import pandas as pd
import numpy as np
from datetime import datetime

# Ensure the CSV file for StudentDetails exists
csv_file = "StudentDetails.csv"

if not os.path.exists(csv_file):
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Roll No", "Name"])  # Column headers

# Initialize the main window
window = tk.Tk()
window.title("Face Recognition Attendance System")
window.geometry('1366x768')
window.configure(bg="#f0f0f0")

# Header
header = tk.Label(
    window, 
    text="Face Recognition Attendance System", 
    bg="#2c3e50", 
    fg="white", 
    font=("Helvetica", 24, "bold"), 
    pady=10
)
header.pack(fill="x")

# Main Frame
main_frame = tk.Frame(window, bg="#f0f0f0")
main_frame.pack(pady=20)

# Input Fields
roll_no_label = tk.Label(main_frame, text="Roll No:", font=("Helvetica", 14), bg="#f0f0f0")
roll_no_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")
roll_no_entry = tk.Entry(main_frame, font=("Helvetica", 14))
roll_no_entry.grid(row=0, column=1, padx=10, pady=10)

name_label = tk.Label(main_frame, text="Student Name:", font=("Helvetica", 14), bg="#f0f0f0")
name_label.grid(row=1, column=0, padx=10, pady=10, sticky="w")
name_entry = tk.Entry(main_frame, font=("Helvetica", 14))
name_entry.grid(row=1, column=1, padx=10, pady=10)

# Notification Section
notification_label = tk.Label(main_frame, text="Notification:", font=("Helvetica", 14, "bold"), bg="#f0f0f0")
notification_label.grid(row=2, column=0, padx=10, pady=10, sticky="w")
notification_message = tk.Label(main_frame, text="", font=("Helvetica", 12), bg="white", width=30, height=2, anchor="w")
notification_message.grid(row=2, column=1, padx=10, pady=10)

# Attendance Section
attendance_label = tk.Label(main_frame, text="Attendance File:", font=("Helvetica", 14, "bold"), bg="#f0f0f0")
attendance_label.grid(row=3, column=0, padx=10, pady=10, sticky="w")
attendance_message = tk.Label(main_frame, text="", font=("Helvetica", 12), bg="white", width=30, height=2, anchor="w")
attendance_message.grid(row=3, column=1, padx=10, pady=10)

# Button Frame
button_frame = tk.Frame(window, bg="#f0f0f0")
button_frame.pack(pady=20)

def clear_entries():
    roll_no_entry.delete(0, 'end')
    name_entry.delete(0, 'end')
    notification_message.config(text="")
    attendance_message.config(text="")

def take_images():
    roll_no = roll_no_entry.get().strip()
    name = name_entry.get().strip().title()

    # Validate Roll No and Name
    if roll_no.isdigit() and all(char.isalpha() or char.isspace() for char in name):
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        sample_num = 0

        while True:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                sample_num += 1
                # Save face image to the TrainingImage directory
                cv2.imwrite(f"TrainingImage/{name.replace(' ', '_')}.{roll_no}.{sample_num}.jpg", gray[y:y+h, x:x+w])
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.imshow('Taking Images', img)

            if cv2.waitKey(100) & 0xFF == ord('q') or sample_num > 20:
                break

        cam.release()
        cv2.destroyAllWindows()

        # Notify the user
        notification_message.config(text=f"Images saved for Roll No: {roll_no}, Name: {name}")

        # Save roll_no and name to the CSV file
        try:
            with open(csv_file, 'a+', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([roll_no, name])
                print(f"Saved to CSV: Roll No: {roll_no}, Name: {name}")
        except Exception as e:
            notification_message.config(text=f"Error saving to CSV: {e}")

    else:
        notification_message.config(text="Invalid input. Roll No must be numeric, and Name must contain only letters and spaces.")

def train_images():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)

    def get_images_and_labels(path):
        image_paths = [os.path.join(path, f) for f in os.listdir(path)]
        faces = []
        roll_nos = []
        for image_path in image_paths:
            img = Image.open(image_path).convert('L')
            img_np = np.array(img, 'uint8')
            roll_no = int(os.path.split(image_path)[-1].split(".")[1])
            faces.append(img_np)
            roll_nos.append(roll_no)
        return faces, roll_nos

    faces, roll_nos = get_images_and_labels("TrainingImage")
    recognizer.train(faces, np.array(roll_nos))
    recognizer.save("TrainingImageLabel/Trainner.yml")
    notification_message.config(text="Model trained successfully!")

def track_images():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainingImageLabel/Trainner.yml")
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    
    # Load the student details
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        notification_message.config(text="StudentDetails.csv not found!")
        return
    
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Roll No', 'Name', 'Time']
    attendance = pd.DataFrame(columns=col_names)

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            roll_no, confidence = recognizer.predict(gray[y:y+h, x:x+w])
            
            if confidence < 50:
                name = df.loc[df['Roll No'] == roll_no]['Name'].values[0]
                timestamp = datetime.now()
                time = timestamp.strftime('%H:%M:%S')

                # Add entry to attendance DataFrame
                attendance.loc[len(attendance)] = [roll_no, name, time]

                # Display detected name
                cv2.putText(img, f"{name} ({roll_no})", (x, y-10), font, 1, (0, 255, 0), 2)
            else:
                cv2.putText(img, "Unknown", (x, y-10), font, 1, (0, 0, 255), 2)

            # Draw rectangle around the face
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Drop duplicate attendance entries
        attendance = attendance.drop_duplicates(subset=['Roll No'], keep='first')

        # Display the video frame
        cv2.imshow('Tracking Images', img)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

    # Generate the file name based on today's date
    timestamp = datetime.now()
    file_name = f"Attendance_{timestamp.strftime('%d%m%Y')}.csv"
    file_path = f"Attendance/{file_name}"
    
    try:
        os.makedirs("Attendance", exist_ok=True)  # Ensure the Attendance folder exists
        
        # Append to the existing file if it exists
        if os.path.exists(file_path):
            attendance.to_csv(file_path, mode='a', header=False, index=False)
        else:
            attendance.to_csv(file_path, mode='w', header=True, index=False)
        
        attendance_message.config(text=f"{file_name}")
        print(f"Attendance saved to {file_name}")
    except Exception as e:
        notification_message.config(text=f"Error saving attendance: {e}")

# Buttons
tk.Button(button_frame, text="Take Images", command=take_images, font=("Helvetica", 14), bg="#3498db", fg="white", width=15).grid(row=0, column=0, padx=10, pady=10)
tk.Button(button_frame, text="Train Model", command=train_images, font=("Helvetica", 14), bg="#2ecc71", fg="white", width=15).grid(row=0, column=1, padx=10, pady=10)
tk.Button(button_frame, text="Mark Attendance", command=track_images, font=("Helvetica", 14), bg="#f1c40f", fg="black", width=15).grid(row=0, column=2, padx=10, pady=10)
tk.Button(button_frame, text="Clear", command=clear_entries, font=("Helvetica", 14), bg="#95a5a6", fg="white", width=15).grid(row=0, column=3, padx=10, pady=10)
tk.Button(button_frame, text="Quit", command=window.quit, font=("Helvetica", 14), bg="#e74c3c", fg="white", width=15).grid(row=0, column=4, padx=10, pady=10)

# Run the application
window.mainloop()