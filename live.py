import cv2
import mediapipe as mp
import time
import operator
import numpy as np

import glfw
from OpenGL.GL import *

head = 0
wristL = 16
wristR = 15
elbowL = 14
elbowR = 13
shoulderL = 12
shoulderR = 11
hipL = 24
hipR = 23
kneeL = 26
kneeR = 25
ankleL = 28
ankleR = 27 

limbWidth = 0.05
scaleFactor = 2
xOffset = 1
yOffset = -0.1

connections = frozenset({(5, 3), (3, 1), (1, 0), (0, 2), (2, 4), (1, 7), 
                        (7, 6), (6, 0), (7, 9), (8, 10), (6, 8), (8, 10)})

class point():
    def __init__(self, x, y):
        self.x = x
        self.y = y

def distance(point1, point2):
    return np.sqrt((point2.x - point1.x)*(point2.x - point1.x) + (point2.y - point1.y) * (point2.y - point1.y))

def drawRect(start, end, width):
    startAdj = point(start.x * scaleFactor, 1 - start.y * scaleFactor)
    endAdj = point(end.x * scaleFactor, 1 - end.y * scaleFactor)

    dX = startAdj.x - endAdj.x
    dY = startAdj.y - endAdj.y
    centreX = (startAdj.x + endAdj.x)/2
    centreY = (startAdj.y + endAdj.y)/2
    length = distance(startAdj, endAdj)
    angle = np.rad2deg(np.arctan2(dY, dX))

    glPushMatrix()

    glTranslated(-xOffset, -yOffset, 0)
    glTranslated(centreX, centreY, 0)
    glRotated(angle, 0, 0, 1)

    glBegin(GL_QUADS)
    glVertex2f(- 0.5 * length, - 0.5 * width)
    glVertex2f(- 0.5 * length, + 0.5 * width)
    glVertex2f(+ 0.5 * length, + 0.5 * width)
    glVertex2f(+ 0.5 * length, - 0.5 * width)
    glEnd()

    glPopMatrix()

def drawCircle(centre, radius):
    twicePi = 2 * 3.1415926535
    numSegments = 20
    
    glPushMatrix()

    glTranslated(-xOffset, -yOffset, 0)
    glTranslated(centre.x * scaleFactor, (1 - centre.y * scaleFactor), 0)
    
    glBegin(GL_TRIANGLE_FAN)
    glVertex2f(0, 0)
    for i in range(numSegments + 1): 
        glVertex2f(radius * np.cos(i *  twicePi / numSegments), radius * np.sin(i * twicePi / numSegments))
    glEnd()

    glPopMatrix()

def drawArms(landmarks):
    drawRect(landmarks[wristL], landmarks[elbowL], limbWidth)
    drawRect(landmarks[elbowL], landmarks[shoulderL], limbWidth)
    drawRect(landmarks[wristR], landmarks[elbowR], limbWidth)
    drawRect(landmarks[elbowR], landmarks[shoulderR], limbWidth)

def drawLegs(landmarks):
    drawRect(landmarks[ankleL], landmarks[kneeL], limbWidth)
    drawRect(landmarks[kneeL], landmarks[hipL], limbWidth)
    drawRect(landmarks[ankleR], landmarks[kneeR], limbWidth)
    drawRect(landmarks[kneeR], landmarks[hipR], limbWidth)

def drawTorsoHead(landmarks):
    torsoStart = point((landmarks[hipR].x + landmarks[hipL].x)/2, (landmarks[hipR].y + landmarks[hipL].y)/2)
    torsoEnd = point((landmarks[shoulderR].x + landmarks[shoulderL].x)/2, (landmarks[shoulderR].y + landmarks[shoulderL].y)/2)
    torsoWidth = (distance(landmarks[hipL], landmarks[hipR]) + distance(landmarks[shoulderL], landmarks[shoulderR])) / 2
    drawRect(torsoStart, torsoEnd, torsoWidth)

    headRadius = distance(torsoEnd, landmarks[head]) * 0.7
    drawCircle(landmarks[head], headRadius)


mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if cap.isOpened(): 
    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    
pTime = 0

glfw.init()

window = glfw.create_window(int(width), int(height), "Dance", None, None)
glfw.set_window_pos(window,1200,200)
glfw.make_context_current(window)
glEnableClientState(GL_VERTEX_ARRAY)
glClearColor(0,0,0,0)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 90)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
#    print(results.pose_landmarks)
#    print(mpPose.POSE_CONNECTIONS)
    
    if results.pose_landmarks:
        body = mp.framework.formats.landmark_pb2.NormalizedLandmarkList(landmark = [
            results.pose_landmarks.landmark[11], 
            results.pose_landmarks.landmark[12],
            results.pose_landmarks.landmark[13], 
            results.pose_landmarks.landmark[14], 
            results.pose_landmarks.landmark[15], 
            results.pose_landmarks.landmark[16],
            results.pose_landmarks.landmark[23], 
            results.pose_landmarks.landmark[24], 
            results.pose_landmarks.landmark[25], 
            results.pose_landmarks.landmark[26], 
            results.pose_landmarks.landmark[27], 
            results.pose_landmarks.landmark[28]
      ])
#        mpDraw.draw_landmarks(img, body, connections)
        glClear(GL_COLOR_BUFFER_BIT)
        drawArms(results.pose_landmarks.landmark)
        drawLegs(results.pose_landmarks.landmark)
        drawTorsoHead(results.pose_landmarks.landmark)

        glfw.swap_buffers(window)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)


'''        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w,c = img.shape
#            print(id, lm)
            cx, cy = int(lm.x*w), int(lm.y*h)
            cv2.circle(img, (cx, cy), 5, (255,0,0), cv2.FILLED)'''

