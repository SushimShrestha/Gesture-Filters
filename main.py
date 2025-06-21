import cv2
import mediapipe as mp
import time
import numpy as np


vid= cv2.VideoCapture(0)
if not vid.isOpened():
    print("Video is not captured.")
mp_hands= mp.solutions.hands
hands= mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw= mp.solutions.drawing_utils
filters=[None, "GRAYSCALE", "SEPIA", "NEGETIVE", "BLUR"]
current_filter=0 #0 is index/none
last_action_time= 0
debounce_time=1 #time between gestures
def applyFilters(frame, filter_type):
    if filter_type=="GRAYSCALE":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif filter_type=="SEPIA":
        sepia_filter= np.array([[0.272,0.534,0.131],
                                [0.349,0.686,0.168],
                                [0.393,0.769,0.189]])
        sepia_frame=cv2.transform(frame, sepia_filter)
        sepia_frame=np.clip(sepia_frame,0,255)
        return sepia_frame.astype(np.uint8)#unsigned integer
    elif filter_type=="NEGETIVE":
        return cv2.bitwise_not(frame)
    elif filter_type=="BLUR":
        return cv2.GaussianBlur(frame,(15,15),0) 
    return  frame   


while True:
    ret, frame = vid.read()
    if ret==False:
        print("Unable to read video.")
    frame=cv2.flip(frame,1)
    frame_rgb= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results= hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            thumb_tip=hand_landmarks.landmark[mp_hands.Hand_Landmark.THUMP_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.Hand_Landmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.Hand_Landmark.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks.landmark[mp_hands.Hand_Landmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.Hand_Landmark.PINKY_FINGER_TIP]
            frame_height, frame_width, _=frame.shape

            thumbx, thumby=int(thumb_tip.x *frame_width), (thumb_tip.y* frame_height)
            indexx, indexy=int(index_tip.x *frame_width), (index_tip.y* frame_height)
            ringx, ringy=int(ring_tip.x *frame_width), (ring_tip.y* frame_height)
            middlex, middley=int(middle_tip.x *frame_width), (middle_tip.y* frame_height)
            pinkyx, pinkyy=int(pinky_tip.x *frame_width), (pinky_tip.y* frame_height)

            cv2.circle(frame, (thumbx,thumby), 10,(255,0,0), cv2.FILLED)
            cv2.circle(frame, (indexx, indexy), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(frame, (middlex, middley), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(frame, (ringx, ringy), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(frame, (pinkyx, pinkyy), 10, (255, 0, 0), cv2.FILLED)
            
            current_time= time.time()
            if abs(thumbx-indexx)<30 and abs(thumby-indexy)<30:
                if (current_time- last_action_time) > debounce_time:
                    cv2.putText(frame,"Picture Captured", (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)
                    last_action_time=current_time
                    cv2.imwrite(f"Picture_{int(time.time())}.jpg", frame)
                    print("Picture Saved!")
                elif (abs(thumbx-middlex)<30 and abs(thumby-middley)<30 or \
                    abs(thumbx-ringx)<30 and abs(thumby-ringy)<30 or \
                    abs(thumbx-pinkyx)<30 and abs(thumby-pinkyy)<30) :
                    if (current_time- last_action_time) > debounce_time:
                        current_filter=(current_filter+1)%len(filter)
                        last_action_time=current_time
                        print(f"Switched to filter: {filters[current_filter]}")
                        
    filter_img=applyFilters(frame,filters[current_filter])
    if filters[current_filter]=="GRAYSCALE":
        cv2.imshow("Gesture Control Photo App", cv2.cvtColor(filter_img, cv2.COLOR_GRAY2BGR))
    else:
        cv2.imshow("Gesture Control photo app", filter_img)           
    if cv2.waitKey(1) & 0xFF== ord("q"):
       break
vid.release()
cv2.destroyAllWindows()    
