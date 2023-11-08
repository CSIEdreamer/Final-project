#!/usr/bin/env python3
print("Import...")
import argparse
import time
import RPi.GPIO as GPIO
import cv2
import numpy as np
import threading
from tensorflow.keras.models import model_from_json
from AlphaBot import AlphaBot
from Infrared_Line_Tracking import TRSensor
from openvino.inference_engine import IECore

class ipcamCapture:
    def __init__(self):
        self.Frame = []
        self.status = False
        self.isstop = False
        
    # 攝影機連接。
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)

    def start(self):
    # 把程式放進子執行緒，daemon=True 表示該執行緒會隨著主執行緒關閉而關閉。
        print('ipcam started!')
        threading.Thread(target=self.queryframe, daemon=True, args=()).start()

    def stop(self):
    # 記得要設計停止無限迴圈的開關。
        self.isstop = True
        print('ipcam stopped!')

    def getframe(self):
    # 當有需要影像時，再回傳最新的影像。
        return self.Frame.copy()
        
    def queryframe(self):
        while (not self.isstop):
            self.status, self.Frame = self.capture.read()
        
        self.capture.release()
        
CS = 5
Clock = 25
Address = 24
DataOut = 23

class_id = 3
is_people = False
stop_line = False
start = False

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(Clock,GPIO.OUT)
GPIO.setup(Address,GPIO.OUT)
GPIO.setup(CS,GPIO.OUT)
GPIO.setup(DataOut,GPIO.IN,GPIO.PUD_UP)

TR = TRSensor()
Ab = AlphaBot()
Ab.stop()
ipcam = ipcamCapture()
ipcam.start()

# 設定程式參數
arg_parser = argparse.ArgumentParser(description='執行模型，辨識影片檔或攝影機影像。')
arg_parser.add_argument(
    '--model-sign',
    required=True,
    help='模型架構檔',
)
arg_parser.add_argument(
    '--weights-sign',
    required=True,
    help='模型參數檔',
)
arg_parser.add_argument(
    '--model-people',
    required=True,
    help='模型架構檔',
)
arg_parser.add_argument(
    '--weights-people',
    required=True,
    help='模型參數檔',
)
arg_parser.add_argument(
    '--input-width',
    type=int,
    default=48,
    help='模型輸入影像寬度',
)
arg_parser.add_argument(
    '--input-height',
    type=int,
    default=48,
    help='模型輸入影像高度',
)
arg_parser.add_argument(
    '--gui',
    action='store_true',
    help='?用圖像界面',
)

# 設置 Movidius 裝置
plugin = IECore()

args = arg_parser.parse_args()
assert args.input_width > 0 and args.input_height > 0

# 載入模型檔
print("Load people model...")
net_people = plugin.read_network(model=args.model_people, weights=args.weights_people)
input_blob_people = next(iter(net_people.input_info))
out_blob_people = next(iter(net_people.outputs))
exec_net_people = plugin.load_network(network=net_people, device_name="MYRIAD", num_requests=2)
print("Load sign model...")
net_sign = plugin.read_network(model=args.model_sign, weights=args.weights_sign)
input_blob_sign = next(iter(net_sign.input_info))
out_blob_sign = next(iter(net_sign.outputs))
exec_net_sign = plugin.load_network(network=net_sign, device_name="MYRIAD", num_requests=2)
print("Load finished\nStart to detect")

def sharpen(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #銳化
    dst = cv2.filter2D(image, -1, kernel=kernel)

    return dst


def tracking():
    global class_id
    global stop_line
    global is_people
    global start
    global TR
    global Ab
    maximum = 45
    integral = 0
    last_proportional = 0
    
    print("Line follow start")
    time.sleep(0.5)
    
    TR.calibratedMin = [775, 925, 819, 971, 965]
    TR.calibratedMax = [63, 88, 114, 111, 92]
    print(TR.calibratedMin)
    print(TR.calibratedMax)
    time.sleep(1)
    Ab.stop()
    try:
        while True:
	        if start:
	            #print(start)
	            while(is_people): #people
	                print("OH! watch out!")
	                Ab.stop()
	                time.sleep(3)
	
	            Ab.backward()		 	 
	            stop_line = TR.readStopLine()
	            if(stop_line):
	                Ab.stop()
	                time.sleep(1)		  
	                print("Stop to detect")
	                if(class_id == 0): #left
	                    Ab.setPWMA(70) 	
	                    Ab.setPWMB(70) 
	                    Ab.left_T()
	                    print("Left")
	                    #time.sleep(2)
	                    while(not TR.readContinue(white_line = 0, left=1)):#tuen left, stop when read continue line
	                        Ab.left_T()
	                    print("Left Done")					   							  	   		
	                    Ab.backward()
	                    #time.sleep(1)
	                elif(class_id == 1): #right
	                    Ab.setPWMA(80) 	
	                    Ab.setPWMB(80)  
	                    Ab.right_T()
	                    print("Right")
	                    #time.sleep(2)
	                    while(not TR.readContinue(white_line = 0, left=0)):#tuen right, stop when read continue line
	                        Ab.right_T()
	                    print("Right Done")	
	                    Ab.backward()
	                    #time.sleep(1)
	                elif(class_id == 2): #stop
	                    Ab.stop()
	                    raise Exception					                    
	                elif(class_id == 3): #other
	                    #time.sleep(0.5)
	                    Ab.setPWMA(50) 	
	                    Ab.setPWMB(50)
	                    Ab.backward()
	                    time.sleep(0.25)
	                stop_line = False
	            
	            position = TR.readLine()
	            #print(position)
	            
	            # The "proportional" term should be 0 when we are on the line.
	            proportional = position - 2000
	            
	            # Compute the derivative (change) and integral (sum) of the position.
	            derivative = proportional - last_proportional
	            integral += proportional
	            
	            # Remember the last position.
	            last_proportional = proportional
	        
	            '''
	            // Compute the difference between the two motor power settings,
	            // m1 - m2.  If this is a positive number the robot will turn
	            // to the right.  If it is a negative number, the robot will
	            // turn to the left, and the magnitude of the number determines
	            // the sharpness of the turn.  You can adjust the constants by which
	            // the proportional, integral, and derivative terms are multiplied to
	            // improve performance.
	            '''
	            power_difference = proportional/25 + derivative/100 #+ integral/1000;  
	    
	            if (power_difference > maximum):
	                power_difference = maximum
	            if (power_difference < - maximum):
	                power_difference = - maximum
	            #print(position,power_difference)
	            if (power_difference < 0):
	                Ab.setPWMB(maximum + power_difference)
	                Ab.setPWMA(maximum)
	            else:
	                Ab.setPWMB(maximum)
	                Ab.setPWMA(maximum - power_difference)
        Ab.stop()
    except Exception:
        Ab.setPWMA(0) 	
        Ab.setPWMB(0)
        Ab.stop()
        print('循跡中斷')

def detecting():
    global is_people
    global stop_line
    global class_id
    global args
    global ipcam
    global start
    global exec_net_sign
    global input_blob_sign
    global out_blob_sign
    global exec_net_people
    global input_blob_people
    global out_blob_people

    try:
    # 主迴圈
        prev_timestamp = time.time()

        while True:
            orig_image = ipcam.getframe()
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #銳化
            orig_image = cv2.filter2D(orig_image, -1, kernel=kernel)
            orig_image = np.uint8(np.clip((2 * (np.int16(orig_image) - 60) + 50), 0, 255))
            #orig_image = image_white_balance(orig_image)
            #ret, orig_image = video_dev.read()
            curr_time = time.localtime()
            # 顯示圖片
            if args.gui:
                cv2.namedWindow("show",0)
                cv2.resizeWindow("show", 320, 320)
                cv2.imshow("show", orig_image)
                cv2.waitKey(1)
            # 縮放?模型輸入的維度、調整數字範圍? 0～1 之間的數值
            preprocessed_image = cv2.resize(
                orig_image.astype(np.float32),
                (args.input_width, args.input_height),
            ) / 255.0
            # 這步驟打包圖片成大小? 1 的 batch
            batch = np.expand_dims(
                np.transpose(preprocessed_image, (2, 0 ,1)),  # 將維度順序從 NHWC 調整? NCHW
                0,
            )
            # 執行預測

            if stop_line:
                is_people = False
                # 執行預測
                request_handle = exec_net_sign.start_async(
                    request_id=0,
                    inputs={input_blob_sign: batch}
                )
                status = request_handle.wait()
                result_batch = request_handle.outputs[out_blob_sign]
                result_onehot = result_batch[0]
                
                left_score, right_score, stop_score, other_score = result_onehot
                class_id = np.argmax(result_onehot)

                if class_id == 0:
                    class_str = 'left'
                elif class_id == 1:
                    class_str = 'right'
                elif class_id == 2:
                    class_str = 'stop'
                if class_id == 3:
                    class_str = 'other'
                # 計算執行時間
                recent_timestamp = time.time()
                period = recent_timestamp - prev_timestamp
                prev_timestamp = recent_timestamp
        
                print('時間：%02d:%02d:%02d ' % (curr_time.tm_hour, curr_time.tm_min, curr_time.tm_sec))
                print('輸出：%.2f %.2f %.2f %.2f' % (left_score, right_score, stop_score, other_score))
                print('類別：%s' % class_str)
                print('費時：%f' % period)
                print('----------------')
                if (class_id == 2): 
                    raise KeyboardInterrupt  
            else: 
                request_handle = exec_net_people.start_async(
                    request_id=0,
                    inputs={input_blob_people: batch}
                )
                status = request_handle.wait()
                result_batch = request_handle.outputs[out_blob_people]
                result_onehot = result_batch[0]
                
                other_score, people_score = result_onehot
                class_id_people = np.argmax(result_onehot)

                if class_id_people == 0:
                    class_str = 'other'
                    is_people = False
                elif class_id_people == 1:
                    class_str = 'people'
                    is_people = True
                # 計算執行時間
                recent_timestamp = time.time()
                period = recent_timestamp - prev_timestamp
                prev_timestamp = recent_timestamp

                #print('時間：%02d:%02d:%02d ' % (curr_time.tm_hour, curr_time.tm_min, curr_time.tm_sec))
                #print('輸出：%.2f %.2f %.2f %.2f' % (left_score, right_score, stop_score, other_score))
                print('輸出：%.2f %.2f' % (other_score, people_score))
                print('類別：%s' % class_str)
                print('費時：%f' % period)
                print('----------------')
            if not start:
                start = True

    except KeyboardInterrupt:
        Ab.setPWMA(0) 	
        Ab.setPWMB(0)
        Ab.stop()
        print('辨識中斷')
    #video_dev.release()
        ipcam.stop()



if __name__ == '__main__':
    detect = threading.Thread(target= detecting, daemon=True, args=())
    detect.start()
    tracking()
    detect.join()
    print("End")