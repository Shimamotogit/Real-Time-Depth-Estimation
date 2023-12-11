from yolov7 import *
from midas import *
from time import sleep

class detection():

    def __init__(self):

        self.model_yolov7 = compile_yolov7()
        self.model_midas, self.network_image_height, self.network_image_width, self.output_key = compile_midas()

        self.DEVICE_ID = 0
        self.capture = cv2.VideoCapture(self.DEVICE_ID)#("D:/Shorts.mp4")  "D:/midas_yolo_test_video.mp4"

    def set_capture(self, WHIDTH:int = 1920, HEIGHT:int = 1080):

        self.SET_WIDTH = WHIDTH
        self.SET_HEIGHT = HEIGHT

        if self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4')):
            print("コーデックをH.264に設定しました。")
        else:
            print("コーデックをH.264に設定できませんでした。")

        if self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.SET_WIDTH):
            print("フレームの幅を1280に設定しました。")
        else:
            print("フレームの幅を1280に設定できませんでした。")

        if self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.SET_HEIGHT):
            print("フレームの高さを720に設定しました。")
        else:
            print("フレームの高さを720に設定できませんでした。")

        if self.capture.set(cv2.CAP_PROP_FPS, 30):
            print("フレームレートを30に設定しました。")
        else:
            print("フレームレートを30に設定できませんでした。")


        self.WHIDTH = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.HEIGHT = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

        return self.WHIDTH, self.HEIGHT


    def detection_image(self, target_names:List[str] = []):
        ret, frame = self.capture.read()

        if ret:

            resized_image = cv2.resize(src=frame, dsize=(self.network_image_height, self.network_image_width))
            input_image = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0)
            result = self.model_midas([input_image])[self.output_key]
            result_image = convert_result_to_image(result=result)

            midas_resized_image = cv2.resize(result_image, (int(self.WHIDTH), int(self.HEIGHT)))
            midas_frame_image = to_gray(midas_resized_image)

            boxes, image, input_shape = detect(self.model_yolov7, frame)
            image_with_boxes = draw_boxes(midas_frame_image, boxes[0], input_shape, image, NAMES, COLORS, target_names=target_names)
            return image_with_boxes

            # cv2.imshow('MiDaS', cv2.resize(midas_resized_image, (int(WHIDTH/2), int(HEIGHT/2))))

            # cv2.imshow('YOLOv7', cv2.resize(image_with_boxes, (int(WHIDTH/2), int(HEIGHT/2))))#cv2.resize(image_with_boxes, (int(WHIDTH), int(HEIGHT))))

            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        else:
            print('フレームを取得できません。')
            time.sleep(1)






