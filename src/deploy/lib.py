import random
import numpy as np
import cv2
import dlib
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from emotions import smiling_detection
from detect_blinks import detect_blinks
from statistics import mode
from imutils import face_utils


frame_window = 10
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
emotion_window = []
ACTION = ["blinking your eyes", "smile"]
distant = ["closer","","far"]
def random_action(num_action = 3):
    lst = []
    for i in range(0, num_action):
        lst.append(random.choice(ACTION))

    if lst[0] == lst[1] == lst[2]:
        if lst[1] == "smile":
            lst[1] = "blinking your eyes"
        else:
            lst[1] = "smile"
    return lst

def distant_face2cam(rect, screen_w, screen_h):
    s_rect = dlib.rectangle.area(rect)
    area_screen = screen_w*screen_h
    r = s_rect/area_screen
    if r < 0.2:
        return 0
    if r < 0.4:
        return 1
    return 2

def show_time(frame, current, total):
    h = frame.shape[0]
    w = frame.shape[1]
    d = int((current*1.0/total)*w)
    frame = cv2.rectangle(frame, (5, h-30), (d, h-10), (0, 255, 0), -1)
    return frame
def smile_detection(faces, gray,smile):
    smile_frame = 10
    smiling = False
    emotion_text, emotion_probability = smiling_detection(faces, gray)
    if len(emotion_text) !=1:
        return smile, emotion_text, emotion_probability, smiling
    if emotion_text[0] == "smile":
        smile["counter_smile"] += 1
    else:
        smile["counter_no_smile"] += 1
        if smile["counter_smile"] > smile_frame and smile["counter_no_smile"] > 10:
            smile["total"] += 1
            smiling = True
            smile["counter_smile"] = 0
            smile["counter_no_smile"] = 0
    return smile, emotion_text, emotion_probability,smiling

def main_detect(frame, eyes_blinking,smile, face_login, action = None,show=False,num=2):
    accept = False
    action_real = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # faces = detector(rgb_image)
    # print(faces)
    _faces = face_login.get_faces([rgb_image])
    faces = dlib.rectangles()

    for face in _faces[0]:

        faces.append(dlib.rectangle(face[0], face[1], face[2], face[3]))
    if len(faces) < 1:
        cv2.putText(rgb_image, "No Face", (frame.shape[0] - 100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        frame = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        return frame, eyes_blinking,smile, accept
    if len(faces) > 1:
        cv2.putText(rgb_image, "Multi Face", (frame.shape[0] - 100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        frame = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        return frame, eyes_blinking,smile, accept

    eyes_blinking, lst_leftEyeHull, lst_rightEyeHull, blinking = detect_blinks(faces,predictor,gray,eyes_blinking)
    if blinking:
        action_real.append("blinking your eyes")
    smile, emotion_text, emotion_probability,smiling = smile_detection(faces,gray,smile)
    if smiling:
        action_real.append("smile")
    for i in range(0, len(emotion_text)):
        dist = distant_face2cam(faces[i], frame.shape[0], frame.shape[1])
        if dist != 1:
            cv2.putText(rgb_image, "Please move {}".format(distant[dist]), (frame.shape[0]-100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        emotion_window.append(emotion_text[i])
        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue

        if emotion_text[i] == 'smile':
            color = emotion_probability[i] * np.asarray((255, 255, 0))
        else:
            color = emotion_probability[i] * np.asarray((255, 0, 0))

        color = color.astype(int)
        color = color.tolist()
        draw_bounding_box(face_utils.rect_to_bb(faces[i]), rgb_image, color)
        if show:
            draw_text(face_utils.rect_to_bb(faces[i]), rgb_image, emotion_mode,
                      color, 0, -45, 1, 1)
    if show:
        cv2.drawContours(rgb_image, lst_leftEyeHull, -1, (0, 255, 0), 1)
        cv2.drawContours(rgb_image, lst_rightEyeHull, -1, (0, 255, 0), 1)
    if action != None and show:
        cv2.putText(rgb_image, "Please {} {} time".format(action,num), (int(frame.shape[0]/2-150),int(frame.shape[1]/2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2)
    frame = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    if len(action_real) == 1:
        if action == action_real[0]:
            accept = True
    return frame, eyes_blinking, smile,accept
