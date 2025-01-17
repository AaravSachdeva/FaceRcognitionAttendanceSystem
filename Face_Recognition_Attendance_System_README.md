
# Face Recognition Attendance System

## Overview
The **Face Recognition Attendance System** is a Python-based application that uses OpenCV, Tkinter, and other libraries to take images of students, train a machine learning model on the images, and track student attendance based on face recognition. It allows users to:

1. Capture face images of students and save them.
2. Train a model to recognize faces.
3. Track student attendance and log it in a CSV file.

## Features
- **Capture Student Faces**: Takes images of students for training the face recognition model.
- **Train Model**: Trains the model using the images captured, allowing the system to recognize students' faces.
- **Mark Attendance**: Marks student attendance by recognizing faces during real-time video capture.
- **Save Attendance**: Saves attendance data in a CSV file.
- **GUI Interface**: The application uses Tkinter for a user-friendly interface.
- **Notification System**: Displays real-time notifications to the user on various actions, such as saving images or training the model.

## Requirements
Before running the application, make sure you have the following Python libraries installed:

- `opencv-python`
- `Pillow`
- `pandas`
- `numpy`
- `tkinter` (Usually pre-installed with Python)
- `csv`
- `datetime`

To install the required libraries, run:

```bash
pip install opencv-python Pillow pandas numpy
```

## Files
1. **StudentDetails.csv**: Stores student roll numbers and names.
2. **TrainingImage/**: Directory where student face images are saved.
3. **TrainingImageLabel/Trainner.yml**: Saved machine learning model for face recognition.
4. **Attendance/**: Directory where attendance logs are saved.
5. **haarcascade_frontalface_default.xml**: Pre-trained classifier for face detection (downloadable from OpenCV repository).

## Usage Instructions

1. **Start the Application**:
   Run the script to launch the Face Recognition Attendance System.

   ```bash
   python attendance_system.py
   ```

2. **Capture Student Faces**:
   - Enter the **Roll No** and **Student Name** in the provided fields.
   - Click on **Take Images**.
   - The system will open a webcam window, and the student will need to stay in front of the camera for face detection.
   - Once enough images are captured, the system will notify the user.

3. **Train the Model**:
   - Click on **Train Model** to train the face recognition model using the captured images.
   - The model will be saved as `Trainner.yml` for future use.

4. **Mark Attendance**:
   - Click on **Mark Attendance** to start face recognition.
   - The system will use the webcam to detect faces in real time and match them with the trained model.
   - Attendance will be saved in a timestamped CSV file under the `Attendance/` folder.

5. **Clear Fields**:
   - Click on **Clear** to reset the input fields and messages.

6. **Exit**:
   - Click on **Quit** to close the application.

## Directory Structure

```
- Face_Recognition_Attendance_System/
    - StudentDetails.csv              # Stores student details (Roll No and Name)
    - TrainingImage/                  # Captured images of students' faces
    - TrainingImageLabel/             # Folder where the trained model (Trainner.yml) is stored
    - Attendance/                     # Folder to store the attendance logs
    - haarcascade_frontalface_default.xml  # Pre-trained face detection model (downloadable from OpenCV)
    - attendance_system.py            # Python script to run the system
```

## Notes
- **Roll No** should be numeric, and the **Name** should only contain letters and spaces.
- The system uses **LBPH (Local Binary Pattern Histogram)** face recognition for training and prediction.
- The system assumes a webcam is available for face capture and recognition.
- The application saves attendance in a CSV file named `Attendance_DDMMYYYY.csv`.

## Troubleshooting
- Ensure that the webcam is accessible and working properly.
- If the `haarcascade_frontalface_default.xml` file is missing, download it from [here](https://github.com/opencv/opencv/tree/master/data/haarcascades).
- If the **StudentDetails.csv** file is not found, the system will display an error message. Ensure the file is in the same directory as the script.
