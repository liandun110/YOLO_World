"""
在图像预处理上，与simple_pt.py完全一致，但是本代码不依赖torch包。
"""
import cv2
import onnxruntime as ort
import numpy as np

def _get_rescale_ratio(old_size, scale):
    w, h = old_size
    max_long_edge = max(scale)
    max_short_edge = min(scale)
    scale_factor = min(max_long_edge / max(h, w), max_short_edge / min(h, w))
    return scale_factor

def letter_resize(image, scale):
    image_shape = image.shape[:2]  # height, width
    ratio = min(scale[0] / image_shape[0], scale[1] / image_shape[1])
    ratio = min(ratio, 1.0)
    ratio = [ratio, ratio]
    no_pad_shape = (int(round(image_shape[0] * ratio[0])), int(round(image_shape[1] * ratio[1])))

    # padding height & width
    padding_h, padding_w = scale[0] - no_pad_shape[0], scale[1] - no_pad_shape[1]

    scale_factor = (no_pad_shape[1] / image_shape[1], no_pad_shape[0] / image_shape[0])

    # padding
    top_padding, left_padding = int(round(padding_h // 2 - 0.1)), int(round(padding_w // 2 - 0.1))
    bottom_padding = padding_h - top_padding
    right_padding = padding_w - left_padding

    padding_list = [top_padding, bottom_padding, left_padding, right_padding]

    pad_val = tuple(114 for _ in range(image.shape[2]))

    padding = (padding_list[2], padding_list[0], padding_list[3], padding_list[1])

    img = cv2.copyMakeBorder(image, padding[1], padding[3], padding[0], padding[2], cv2.BORDER_CONSTANT, value=pad_val)
    return img

image = cv2.imread('/home/suma/projects/app/png1.jpg')

original_h, original_w = image.shape[:2]
scale = (640, 640)
ratio = _get_rescale_ratio((original_h, original_w), scale)

image = cv2.resize(image, dsize=(int(original_w * ratio), int(original_h * ratio)), interpolation=3)
print('缩放图像，得到尺寸为：', image.shape[:2])

resized_h, resized_w = image.shape[:2]
scale_ratio_h = resized_h / original_h
scale_ratio_w = resized_w / original_w
scale_factor = (scale_ratio_w, scale_ratio_h)

image = letter_resize(image, scale)
print('经过letter resize后的图像尺寸为：', image.shape[:2])

# 将BGR图像转换为RGB图像
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 将图像数据标准化到[0, 1]范围内
image_norm = image_rgb / 255.0

# 增加一个维度以匹配模型输入
inputs = np.expand_dims(image_norm, axis=0)
inputs = inputs.astype(np.float32).transpose((0, 3, 1, 2))

# 确保输入数据类型为float32
inputs = inputs.astype(np.float32)

# 加载ONNX模型
session = ort.InferenceSession("/home/suma/projects/app/weights/yolow-l.onnx")

# 获取输入名称
input_name = session.get_inputs()[0].name

# 运行模型推理
result = session.run(None, {input_name: inputs})

print(result)

# 绘制检测结果
# boxes = result[1][0]  # ndarray, shape=(num_obj, 4)，检测出的边框坐标，x1, y1, x2, y2
# for box in boxes:
#     print(box)
#     x1, y1, x2, y2 = box
#     color = (0, 255, 0)  # 绿色
#     thickness = 2
#     cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
# cv2.imwrite('output_with_boxes.png', image)  # 保存图像

# 绘制检测结果
boxes = result[1][0]  # ndarray, shape=(num_obj, 4)，检测出的边框坐标，x1, y1, x2, y2
confidences = result[2][0]  # 每个框的置信度
class_ids = result[3][0]  # 每个框对应的类别ID

for i, box in enumerate(boxes):
    x1, y1, x2, y2 = box
    confidence = confidences[i]  # 当前框的置信度
    class_id = class_ids[i]  # 当前框的类别ID

    # 设置框的颜色（绿色）
    color = (0, 255, 0)
    thickness = 2

    # 绘制框
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

    # 在框上方绘制类别ID和置信度
    label = f"ID: {class_id}, Conf: {confidence:.2f}"
    cv2.putText(image, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# 保存带框和标签的图像
cv2.imwrite('output_with_boxes_and_labels.png', image)

