import cv2 as cv
from ultralytics import YOLO 

video = cv.VideoCapture('../datasets/carDetection/highway.mp4')
#video = cv.VideoCapture('../datasets/carDetection/inside.mp4')

coco_model = YOLO("yolov8m.pt")
frame_temp = 0
vehicles = [2, 3, 5, 7]

while video.isOpened():
    ret, frame = video.read()
    frame_temp +=1 
    if ret == True:
        if frame_temp < 300:
            detections = coco_model(frame)[0]
            detections_ = []
            print('====Frame==== ', frame_temp)
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                print('detection:')
                print(x1,'|', y1,'|', x2,'|', y2, '-', score,'-', class_id)
                if int(class_id) in vehicles:
                    detections_.append([x1, y1, x2, y2, score])
                    cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),(0, 255, 0), 1) #work  
                    cv.imshow('videoCar', frame)
                print('detections_:')
                print('cantidad de autos en video: ', len(detections_)) 
                print(detections_)
                print('=============')
        #cv.rectangle(frame, (552, 263), (609, 310), (0, 255, 0), 2) #draw its ok 
        cv.imshow('videoCar', frame)
        
        if cv.waitKey(25) == ord('q'):
            break

video.release()
cv.destroyAllWindows()





















