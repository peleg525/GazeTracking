"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2
from gaze_tracking import GazeTracking
from PIL import Image
import numpy

gaze = GazeTracking()
#webcam = cv2.VideoCapture(0)

# We get a new frame from the webcam

frame = Image.open(r"C:\Users\דגנית\Desktop\PYTHON\eye\GazeTracking\gaze_tracking\trained_models\eb477624-fa4c-4e62-ba81-598d7921573d.jpg").convert('RGB') 
frame = numpy.array(frame) 

# We send this frame to GazeTracking to analyze it
gaze.refresh(frame)

frame = gaze.annotated_frame()
text = ""

if gaze.is_blinking():
    text = "Blinking"
elif gaze.is_right():
    text = "Looking right"
elif gaze.is_left():
    text = "Looking left"
elif gaze.is_center():
    text = "Looking center"

#cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

left_pupil = gaze.pupil_left_coords()
right_pupil = gaze.pupil_right_coords()
cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

cv2.imwrite('savedImage.jpg', frame) 
