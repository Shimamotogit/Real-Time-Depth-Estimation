import numpy as np
import torch
from utils.datasets import letterbox
from typing import List, Tuple, Dict
from utils.general import scale_coords, non_max_suppression
from openvino.runtime import Model
from openvino.runtime import Core
import cv2

def preprocess_image(img0: np.ndarray):

    img = letterbox(img0, auto=False)[0]

    # Convert
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    return img, img0

def prepare_input_tensor(image: np.ndarray):

    input_tensor = image.astype(np.float32)
    input_tensor /= 255.0

    if input_tensor.ndim == 3:
        input_tensor = np.expand_dims(input_tensor, 0)
    return input_tensor

NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush']

COLORS = {name: [np.random.randint(0, 255) for _ in range(3)]
          for i, name in enumerate(NAMES)}

def detect(model: Model, image_path: str = "Path", conf_thres: float = 0.25, iou_thres: float = 0.45, classes: List[int] = None, agnostic_nms: bool = False):

    output_blob = model.output(0)

    preprocessed_img, orig_img = preprocess_image(image_path)
    input_tensor = prepare_input_tensor(preprocessed_img)
    predictions = torch.from_numpy(model(input_tensor)[output_blob])
    pred = non_max_suppression(predictions, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
    return pred, orig_img, input_tensor.shape

def depth_info_convert(depth_info_lst, px_value, coordinate_value):
    
    if depth_info_lst == []:
        depth_info_lst.append([px_value, coordinate_value])
        return depth_info_lst

    for i in range(len(depth_info_lst)):
        
        if depth_info_lst[i][0] > px_value:

            if i+1 == len(depth_info_lst):
                depth_info_lst.append([px_value, coordinate_value])
                return depth_info_lst
            continue

        elif depth_info_lst[i][0] <= px_value:
            depth_info_lst.insert(i, [px_value, coordinate_value])
            print(f"insert : {depth_info_lst}")
            return depth_info_lst

        else:
            print("lst_convert_error")

def draw_boxes(midas_frame_image: np.ndarray, predictions: np.ndarray, input_shape: Tuple[int], image: np.ndarray, names: List[str], colors: Dict[str, int], target_names: List[str]):
    
    depth_info_lst = []
    if not len(predictions):
        return image

    predictions[:, :4] = scale_coords(input_shape[2:], predictions[:, :4], image.shape).round()

    for *xyxy, conf, cls in reversed(predictions):
        label = f'{names[int(cls)]} {conf:.2f}'

        if label.split(" ")[0] in target_names:

            center_x = int(int(xyxy[0])+((int(xyxy[2])-int(xyxy[0]))/2))
            center_y = int(int(xyxy[1])+((int(xyxy[3])-int(xyxy[1]))/2))

            label_values = [midas_frame_image[center_y, center_x], [center_x, center_y]]
            depth_info_lst = depth_info_convert(depth_info_lst, label_values[0], label_values[1])

    # print(f"depth_info_lst : {depth_info_lst}")　#座標の表示
    for i, depth_info in enumerate(depth_info_lst, start = 1):

        cv2.putText(image, text = str(i),
                    org = tuple(depth_info[1]), 
                    fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale = 2.0,
                    color = (0, 255, 0),
                    thickness=7,
                    lineType=cv2.LINE_4)
    return image

def compile_yolov7():

    print("\nYOLOv7の準備中...\n")

    core = Core()
    model = core.read_model('./yolov7_file/yolov7/model/yolov7-tiny.xml')

    compiled_model = core.compile_model(model, "GPU")

    print("\nYOLOv7の準備完了\n")

    return compiled_model

