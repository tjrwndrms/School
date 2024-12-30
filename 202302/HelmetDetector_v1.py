import cv2
import os
import numpy as np
import time 
import threading
import time
import datetime
from queue import Queue

from collections import deque
import final_v1 as ui


class CamWrapper(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.rtsp_url = 'rtsp://admin:12abcdefg@192.168.0.99:554/cam/realmonitor?channel=1&subtype=0'
        print(self.rtsp_url)
        self.is_stop = False
        self.is_bind = False
        self.prev_time = None
        self.cam = None
        self.class_count = 0
        
        self.PARSER_FPS = 30 # 사용할 FPS 지정
        self.DISP_FLAG = True # DISPLAY VIEW 사용시 True 미사용시 False
        self.SAVE_FLAG = True # 파일저장 사용시 True 미사용시 False
        
        # yolo v4 - tiny 모델
        weights_path = 'C:/darknet-master/darknet-master/backup/yolov4-tiny-human_last.weights'
        config_path = 'C:/darknet-master/darknet-master/cfg/yolov4-tiny-human.cfg'
        
        # 일반 yolo v4 모델
        weights_path2 = 'C:/darknet-master/darknet-master/backup/yolov4-helmet_last.weights'
        config_path2 = 'C:/darknet-master/darknet-master/cfg/yolov4-helmet.cfg'
        
        labels_path = 'C:/darknet-master/darknet-master/data/HumanClassNames.names'
        labels_path2 = 'C:/darknet-master/darknet-master/data/HelmetClassNames.names'
        self.video_path = 'c:/Users/USER/Videos/clp/clptest2.mp4'


        self.net = cv2.dnn.readNet(weights_path, config_path)
        self.net2= cv2.dnn.readNet(weights_path2,config_path2)
        self.classes = []
        with open(labels_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
            
        self.classes_helmet = []
        with open(labels_path2, 'r') as f:
            self.classes_helmet = [line.strip() for line in f.readlines()]
            
        #헬멧 구분용 클래스 만들기

        self.layer_names = self.net.getUnconnectedOutLayersNames()
        self.layer_names_helmet = self.net2.getUnconnectedOutLayersNames()
        self.class_count =0
        self.captured_image = None

    def run(self):
        fps_delay = 1 / PARSER_FPS
        #cv2.ocl.setUseOpenCL(True)
        frame_interval = 1
        frame_counter = 0
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        print(cv2.cuda.getCudaEnabledDeviceCount())

        while True:
            if self.is_stop:
                time.sleep(2)
                break
            
            if self.is_bind:
                ret, self.frame = self.cam.read()
                if not ret:
                    break
                elif ret:   
                    frame_counter += 1
                    #민석이형 코드에 frame 보내주기
                    if(frame_counter >= frame_interval):
                    
                    #cv2.imshow("Frame", self.frame)
                        #detect_frame = self.detect_human(self.frame)
                        detect_frame = self.detect_human(self.frame)
                        #if 
                        #cv2.imshow("Detector", detect_frame)
                        cv2.imshow("Detector", self.frame)
                        frame_counter =0
                    
                    #객체 검출 코드 기입
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.is_stop = True
            if fps_delay > 0:
                time.sleep(fps_delay)
            else:
                time.sleep(0.002)
            

    def bind(self):
        print('Cam Bind Start')
        #self.cam = cv2.VideoCapture(self.rtsp_url)
        self.cam=cv2.VideoCapture(self.video_path)
        print('cam cam')

        if self.cam.isOpened():
            self.is_bind = True
            print('Cam Bind Sussces')
            
    def detect_human(self, frame):
        height, width, _ = frame.shape

        blob = cv2.dnn.blobFromImage(frame, 0.00392, (608, 352), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.layer_names)

        class_ids = []
        confidences = []
        boxes = []
        
        

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.6:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        #indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        indexes = range(len(boxes))
        if len(indexes) == 0:
        # No object of class 0 detected
            self.class_count = 0
            print("초기화")
        else:
            print(f"Detected object of class 0 with confidence: {confidence}")
            self.class_count +=1
            print(self.class_count)
            if self.class_count >=30:
                print("사람 검출")
                frame= self.detect_helmet(frame)
                #헬멧 호출
                self.class_count = 0
                

        '''for i in range(len(boxes)):
            if i in indexes:
                

                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                color = (0, 255, 0)
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)'''
        
        return frame
    
    def detect_helmet(self, frame):
        height, width, _ = frame.shape

        blob = cv2.dnn.blobFromImage(frame, 0.00392, (608, 352), (0, 0, 0), True, crop=False)
        self.net2.setInput(blob)
        outs = self.net2.forward(self.layer_names_helmet)

        class_ids = []
        confidences = []
        boxes = []
        boxes_head = []
        boxes_helmet = []
        confidence_head = []
        confidence_helmet = []
        
        

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.4:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
                    if(class_id == 0):  
                        boxes_head.append([x, y, w, h])
                        confidence_head.append(float(confidence))
                    elif(class_id == 1):
                        boxes_helmet.append([x, y, w, h])
                        confidence_helmet.append(float(confidence))

        indexes_head = cv2.dnn.NMSBoxes(boxes_head, confidence_head, 0.5, 0.4)
        print(indexes_head)
        indexes_helmet = cv2.dnn.NMSBoxes(boxes_helmet, confidence_helmet, 0.5, 0.4)
        print(indexes_helmet)
        
        #indexes_detecte = range(len(boxes))
        indexes = np.concatenate((indexes_head, indexes_helmet))
        print(indexes)
        for i in range(len(boxes)):
            if i in indexes:

                x, y, w, h = boxes[i]
                label = str(self.classes_helmet[class_ids[i]])
                confidence = confidences[i]
                if class_ids[i]==0:
                    color = (0,0 , 255)
                else:
                    color = (255,0,0)
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                
                print("계산시작")
                if class_ids[i] == 0:
                    helmet_detected = any(class_id == 1 for class_id in class_ids)
                    print(helmet_detected)
                    if helmet_detected:
                        print(len(boxes))
                        helmet_worn=False
                        for j in range(len(boxes)):
                            if j in indexes and class_ids[j] == 1:
                                print("iou계산 시작")
                                iou = calculate_iou(boxes[i], boxes[j])
                                print(iou)
                                if iou >= 0.5:
                                    helmet_worn =True
                                    break
                        if helmet_worn:
                            print("안전모 착용")
                        else:
                            print("안전모 미착용")
                            self.set_captured_image(frame)
                                   
                    else:
                        print("안전모 미인식")
                        self.set_captured_image(frame)
                
        return frame
    def set_captured_image(self, captured_image):
        self.captured_image = captured_image

    def get_captured_image(self):
        return self.captured_image

def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate coordinates of intersection area
    x_intersection = max(x1, x2)
    y_intersection = max(y1, y2)
    w_intersection = max(0, min(x1 + w1, x2 + w2) - x_intersection)
    h_intersection = max(0, min(y1 + h1, y2 + h2) - y_intersection)

    # Calculate area of intersection and union
    area_intersection = w_intersection * h_intersection
    area_union = w1 * h1 + w2 * h2 - area_intersection

    # Calculate IoU
    iou = area_intersection / max(area_union, 1e-6)

    return iou

if __name__ == '__main__':
    print('Program Start')
    
    frame_queue = Queue()
    detector_queue = Queue()

    PARSER_FPS = 30 # 사용할 FPS 지정
    DISP_FLAG = True # DISPLAY VIEW 사용시 True 미사용시 False
    SAVE_FLAG = True # 파일저장 사용시 True 미사용시 False
    
    # yolo v4 - tiny 모델
    weights_path = 'C:/darknet-master/darknet-master/backup/yolov4-tiny-human_last.weights'
    config_path = 'C:/darknet-master/darknet-master/cfg/yolov4-tiny-human.cfg'
    
    # 일반 yolo v4 모델
    weights_path2 = 'C:/darknet-master/darknet-master/backup/yolov4-helmet_last.weights'
    config_path2 = 'C:/darknet-master/darknet-master/cfg/yolov4-helmet.cfg'
    
    labels_path = 'C:/darknet-master/darknet-master/data/HumanClassNames.names'
    labels_path2 = 'C:/darknet-master/darknet-master/data/HelmetClassNames.names'
    video_path = 'c:/Users/USER/Videos/clp/clptest2.mp4'


    net = cv2.dnn.readNet(weights_path, config_path)
    net2= cv2.dnn.readNet(weights_path2,config_path2)
    classes = []
    with open(labels_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
        
    classes_helmet = []
    with open(labels_path2, 'r') as f:
        classes_helmet = [line.strip() for line in f.readlines()]
        
    #헬멧 구분용 클래스 만들기

    layer_names = net.getUnconnectedOutLayersNames()
    layer_names_helmet = net2.getUnconnectedOutLayersNames()
    class_count =0

    cam = CamWrapper()
    cam.start()
    
    cam.bind()