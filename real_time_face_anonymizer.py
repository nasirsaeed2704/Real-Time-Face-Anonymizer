import cv2
import mediapipe as mp

#function to detect faces
def anonymize_face(img):
    mp_face_detection = mp.solutions.face_detection

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out = face_detection.process(rgb_img)

        if out.detections is not None:  #making sure a page is not detected
            for detection in out.detections:
                bbox = detection.location_data.relative_bounding_box    #detecting face
                x, y, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height  #retrieving relative location of face

                #converting to actual location
                x1, y1 = int(x*img.shape[1]), int(y*img.shape[0]) 
                x2, y2 = x1 + int(w*img.shape[1]), y1 + int(h*img.shape[0])

                #blurring the portion of the picture where the face is
                img[y1:y2, x1:x2, :] = cv2.blur(img[y1:y2, x1:x2, :], (50, 50))
    return img


cap = cv2.VideoCapture(0)   #initializing object to capture video from webcam

while True:
    ret, frame = cap.read() #reading a frame of webcam video
    frame = anonymize_face(frame)   #detecting and blurring where the face is in the video
    cv2.imshow('Web Cam', frame)
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()