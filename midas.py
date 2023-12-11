
from pathlib import Path
import cv2
import matplotlib.cm
import numpy as np
from openvino.runtime import Core

model_folder = Path('midas_file/model')
ir_model_name_xml = 'MiDaS_small.xml'
ir_model_name_bin = 'MiDaS_small.bin'
model_xml_path = model_folder / ir_model_name_xml

def normalize_minmax(data):
    """Normalizes the values in `data` between 0 and 1"""
    return (data - data.min()) / (data.max() - data.min())
    # return data / 255


def convert_result_to_image(result, colormap="viridis"):
    """
    Convert network result of floating point numbers to an RGB image with
    integer values from 0-255 by applying a colormap.

    `result` is expected to be a single network result in 1,H,W shape
    `colormap` is a matplotlib colormap.
    See https://matplotlib.org/stable/tutorials/colors/colormaps.html
    """
    cmap = matplotlib.cm.get_cmap(colormap)
    # cmap = cv2.cvtColor(cmap, cv2.COLOR_RGB2GRAY)
    result = result.squeeze(0)
    result = normalize_minmax(result)
    result = cmap(result)[:, :, :3] * 255     
    result = result.astype(np.uint8)
    return result

def to_rgb(image_data) -> np.ndarray:
    """
    Convert image_data from BGR to RGB
    """
    return cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)

def to_gray(image_data) -> np.ndarray:
    """
    Convert image_data from BGR to gray
    """
    return cv2.cvtColor(image_data.copy(), cv2.COLOR_BGR2GRAY)

def compile_midas():
    print("\nMiDaSの準備中...\n")
    core = Core()
    core.set_property({'CACHE_DIR': '../cache'})
    model = core.read_model(model_xml_path)
    compiled_model = core.compile_model(model=model, device_name="GPU")

    input_key = compiled_model.input(0)
    output_key = compiled_model.output(0)

    network_input_shape = list(input_key.shape)
    network_image_height, network_image_width = network_input_shape[2:]
    print("\nMiDaSの準備完了\n")
    
    return compiled_model, network_image_height, network_image_width, output_key

