from yolov7 import *
from midas import *
from time import sleep
model_yolov7 = compile_yolov7()

model_midas, network_image_height, network_image_width, output_key = compile_midas()

DEVICE_ID = 0
SET_WIDTH = int(1920)
SET_HEIGHT = int(1080)
capture = cv2.VideoCapture(DEVICE_ID)#("D:/Shorts.mp4")  "D:/midas_yolo_test_video.mp4"

if capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4')):
    print("コーデックをH.264に設定しました。")
else:
    print("コーデックをH.264に設定できませんでした。")

if capture.set(cv2.CAP_PROP_FRAME_WIDTH, SET_WIDTH):
    print("フレームの幅を1280に設定しました。")
else:
    print("フレームの幅を1280に設定できませんでした。")

if capture.set(cv2.CAP_PROP_FRAME_HEIGHT, SET_HEIGHT):
    print("フレームの高さを720に設定しました。")
else:
    print("フレームの高さを720に設定できませんでした。")
    
if capture.set(cv2.CAP_PROP_FPS, 30):
    print("フレームレートを30に設定しました。")
else:
    print("フレームレートを30に設定できませんでした。")

# FPS = capture.get(cv2.CAP_PROP_FPS)
WHIDTH = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
HEIGHT = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

# WHIDTH = 1280
# HEIGHT = 720

while(True):
    ret, frame = capture.read()
    
    if ret:
        print(frame.shape)
        resized_image = cv2.resize(src=frame, dsize=(network_image_height, network_image_width))
        input_image = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0)

        result = model_midas([input_image])[output_key]
        result_image = convert_result_to_image(result=result)
        midas_resized_image = cv2.resize(result_image, (int(WHIDTH), int(HEIGHT)))
        midas_frame_image = to_gray(midas_resized_image)

        boxes, image, input_shape = detect(model_yolov7, frame)
        image_with_boxes = draw_boxes(midas_frame_image, boxes[0], input_shape, image, NAMES, COLORS, target_names=['person', 'bottle'])
        
        cv2.imshow('MiDaS', cv2.resize(midas_resized_image, (int(WHIDTH/2.5), int(HEIGHT/2.5))))
        cv2.imshow('YOLOv7', cv2.resize(image_with_boxes, (int(WHIDTH/2.5), int(HEIGHT/2.5))))#cv2.resize(image_with_boxes, (int(WHIDTH), int(HEIGHT))))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print('フレームを取得できません。')
        time.sleep(1)

