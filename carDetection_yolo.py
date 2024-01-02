import cv2 as cv
from ultralytics import YOLO 

video = cv.VideoCapture('../datasets/traffic-peru/callao-miraflores.mp4')
coco_model = YOLO("yolov8m.pt")
frame_temp = 0
objects = [0, 2, 3, 5, 7]
#https://docs.ultralytics.com/datasets/detect/coco/#dataset-yaml
names_objects = {0:'person', 2:'car', 3:'motorcycle', 5:'bus', 7:'truck'}
frame_width = int(video.get(3)) 
frame_height = int(video.get(4)) 
size = (frame_width, frame_height)
path_result = '../datasets/traffic-peru/result-callao-miraflores.mp4'
fourcc = cv.VideoWriter_fourcc(*'mp4v')
result = cv.VideoWriter(path_result, fourcc, 30, size)

while video.isOpened():
    ret, frame = video.read()
    if ret == True:
        detections = coco_model(frame)[0]
        #detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            print('data object:')
            print(x1,'|', y1,'|', x2,'|', y2, '-', score,'-', class_id)
            if int(class_id) in objects:
                #detections_.append([x1, y1, x2, y2, score])
                text = names_objects.get(class_id)
                cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),(0, 255, 0), 1) #work
                cv.putText(frame, text, (int(x1 + 5), int(y1 - 8)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  
                result.write(frame)
                cv.imshow('videoCar', frame)
            #print('detections_:')
            #print('cantidad de objetos en frame: ', len(detections_)) 
            #print(detections_)
        
        if cv.waitKey(25) == ord('q'):
            break

video.release()
result.release()
cv.destroyAllWindows()


#while video.isOpened():
#    ret, frame = video.read()
#    frame_temp +=1 
#    if ret == True:
#        if frame_temp < 100:
#            detections = coco_model(frame)[0]
#            detections_ = []
#            print('====Frame==== ', frame_temp)
#            for detection in detections.boxes.data.tolist():
#                x1, y1, x2, y2, score, class_id = detection
#                print('detection:')
#                print(x1,'|', y1,'|', x2,'|', y2, '-', score,'-', class_id)
#                if int(class_id) in objects:
#                    detections_.append([x1, y1, x2, y2, score])
#                    text = names_objects.get(class_id)
#                    cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),(0, 255, 0), 1) #work
#                    cv.putText(frame, text, (int(x1 + 5), int(y1 - 8)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  
#                    cv.imshow('videoCar', frame)
#                print('detections_:')
#                print('cantidad de autos en video: ', len(detections_)) 
#                print(detections_)
#                print('=============')
#        cv.imshow('videoCar', frame)
#        
#        if cv.waitKey(25) == ord('q'):
#            break
#
#video.release()
#cv.destroyAllWindows()





















