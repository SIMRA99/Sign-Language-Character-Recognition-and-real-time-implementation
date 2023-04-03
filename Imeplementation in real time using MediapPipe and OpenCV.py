# Imeplementation in real time using MediapPipe and OpenCV


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# import tensorflow as tf
import cv2
import mediapipe as mp
from keras.models import load_model
import numpy as np
import time
import pandas as pd

my_model = load_model('AlphaTeam_Model')

media_hands = mp.solutions.hands
hand_s = media_hands.Hands()
media_p_drawing = mp.solutions.drawing_utils
my_camera = cv2.VideoCapture(0)

_, frame = my_camera.read()

h, w, c = frame.shape

img_counter = 0
analysis_frames = ''
alphabet_predictions = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
while True:
    _, frame = my_camera.read()

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        analysis_frames = frame
        show_2frame = analysis_frames
        cv2.imshow("Simra_Project", show_2frame)
        frame_rgb_Analysis = cv2.cvtColor(analysis_frames, cv2.COLOR_BGR2RGB)
        result_Analysis = hand_s.process(frame_rgb_Analysis)
        my_hand_landmarks = result_Analysis.multi_hand_landmarks
        if my_hand_landmarks:
            for handLMs in my_hand_landmarks:
                x_max = 0
                y_max = 0
                x_min = w
                y_min = h
                for lmanalysis in handLMs.landmark:
                    x, y = int(lmanalysis.x * w), int(lmanalysis.y * h)
                    if x > x_max:
                        x_max = x
                    if x < x_min:
                        x_min = x
                    if y > y_max:
                        y_max = y
                    if y < y_min:
                        y_min = y
                y_min -= 20
                y_max += 20
                x_min -= 20
                x_max += 20 

        analysisframe = cv2.cvtColor(analysis_frames, cv2.COLOR_BGR2GRAY)
        analysisframe = analysis_frames[y_min:y_max, x_min:x_max]
        analysisframe = cv2.resize(analysis_frames,(28,28))


        nlist = []
        rows,cols = analysisframe.shape
        for i in range(rows):
            for j in range(cols):
                k = analysisframe[i,j]
                nlist.append(k)
        
        datan = pd.DataFrame(nlist).T
        colname = []
        for val in range(784):
            colname.append(val)
        datan.columns = colname

        pixeldata = datan.values
        pixeldata = pixeldata / 255
        pixeldata = pixeldata.reshape(-1,28,28,1)
        prediction = my_model.predict(pixeldata)
        predarray = np.array(prediction[0])
        letter_prediction_dict = {alphabet_predictions[i]: predarray[i] for i in range(len(alphabet_predictions))}
        predarrayordered = sorted(predarray, reverse=True)
        high1 = predarrayordered[0]
        high2 = predarrayordered[1]
        high3 = predarrayordered[2]
        for key,value in letter_prediction_dict.items():
            if value==high1:
                print("Predicted Character 1: ", key)
                print('Confidence 1: ', 100*value)
            elif value==high2:
                print("Predicted Character 2: ", key)
                print('Confidence 2: ', 100*value)
            elif value==high3:
                print("Predicted Character 3: ", key)
                print('Confidence 3: ', 100*value)
        time.sleep(5)

    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hand_s.process(framergb)
    my_hand_landmarks = result.multi_hand_landmarks
    if my_hand_landmarks:
        for handLMs in my_hand_landmarks:
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for lm in handLMs.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
            y_min -= 20
            y_max += 20
            x_min -= 20
            x_max += 20
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.imshow("Simra's Frame", frame)

my_camera.release()
cv2.destroyAllWindows()