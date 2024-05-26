# Importing Libraries
import cv2                        
import mediapipe as mp            
import numpy as np                
import math               

mp_drawing = mp.solutions.drawing_utils    # Assigning drawing_utils from mediapipe as mp_drawing
mp_holistic = mp.solutions.holistic        # Assigning holistic from mediapipe as mp_holistic

def angle_between_lines(x1, y1, x2, y2, x3, y3):         # Defining a function to calculate angle between lines
    # Calculate the slopes of the two lines
    slope1 = (y2 - y1) / (x2 - x1)                       
    slope2 = (y3 - y2) / (x3 - x2)                        
    
    # Calculate the angle between the two lines
    angle = math.atan2(slope2 - slope1, 1 + slope1 * slope2)   # Calculate the angle using the slopes
    
    # Convert the angle to degrees and return it
    return math.degrees(angle)                                # Return the angle in degrees

leglift = 0           # Initialize a variable to count the number of leg lifts
count1 = False        # Initialize a boolean variable to keep track of the first position
count2 = False        # Initialize a boolean variable to keep track of the second position
count3 = False        # Initialize a boolean variable to keep track of the third position

# Initializing the Holistic model with minimum detection and tracking confidence
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:  

    cap = cv2.VideoCapture('resources/side.mp4')    # Start capturing the video from the file "sidelyinglegliftvideo.mp4"
    # cap = cv2.VideoCapture(0)                            # Alternatively, we can capture the video from the webcam (0)
 
    while cap.isOpened():                # While the video is being captured
        ret, frame = cap.read()          # Read the frame from the video

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # Convert the image to RGB

        results = holistic.process(image)    # Make a detection using the Holistic model on the image

        annotated_image = image.copy()       # Make a copy of the image to draw landmarks on

        mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)   # Draw the detected landmarks on the image
        
        left_hip = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP]     # Get the coordinates of the left hip
        right_hip = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP]   # Get the coordinates of the right hip

        midpoint = ((left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2)    # Calculate the midpoint between the left hip and right hip

        left_knee = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_KNEE]    # Get the coordinates
        right_knee = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_KNEE]

        angle1 = angle_between_lines(left_knee.x, left_knee.y, midpoint[0], midpoint[1], right_knee.x, right_knee.y)
        print("Angles :",angle1)
        
        if (angle1 > 60):
            count1 = True
        if (count1 == True and angle1 > 100):
            count2 = True
        if (count2 == True and angle1 < 60):
            count3 = True
        if (count1 == True and count2 == True and count3 == True):
            leglift = leglift + 1
            count1 = False
            count2 = False
            count3 = False
        lg = leglift

        print("Leg Lift : ",leglift)
        # Draw a circle at the midpoint
        cv2.circle(annotated_image, (int(midpoint[0] * annotated_image.shape[1]), int(midpoint[1] * annotated_image.shape[0])), 5, (255, 0, 0), -1)

        # check if angle is between 68.85 to 80 and display "Correct Exercise" on screen
        if 68.85 <= angle1 <= 80:
            cv2.putText(annotated_image, "Correct Side Lying Leg Lift Exercise", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(annotated_image, "Incorrect Side Lying Leg Lift Exercise", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the value of angle1 and leglift on the output screen
        cv2.putText(annotated_image, "Angle: " + str(round(angle1, 2)), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        # Display the value of angle1 and leglift on the output screen
        cv2.putText(annotated_image, "Leg Lift: " + str(round(lg, 2)), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Show the annotated image
        cv2.imshow('MediaPipe Holistic', annotated_image)

        # Exit if the user presses the 'q' key
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()