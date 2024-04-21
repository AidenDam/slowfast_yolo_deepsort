import cv2
import numpy as np

from typing import Tuple

class Draw:
    DEFAULT_COLOR = (0, 255, 0)

    @staticmethod
    def init_colors(class_names, random_seed=8):
        rng = np.random.default_rng(random_seed)
        colors = rng.uniform(0, 255, size=(len(class_names), 3))
        Draw.colors = {key:colors[idx] for idx, key in enumerate(class_names.values())}

    @staticmethod
    def draw_detections(image, boxes, labels, scores, tracked, mask_alpha=0.1):
        det_img = image.copy()

        img_height, img_width = image.shape[:2]
        font_size = min([img_height, img_width]) * 0.0006
        text_thickness = int(min([img_height, img_width]) * 0.001)

        det_img = Draw.draw_masks(det_img, boxes, mask_alpha)

        # Draw bounding boxes and labels of detections
        for i, (label, score, box) in enumerate(zip(labels, scores, boxes)):
            if len(tracked) > 0:
                identity = tracked[tracked[:,0]==i]
                identity = '' if len(identity) == 0 else str(identity[0][-1])
            else:
                identity = ''

            Draw.draw_box(det_img, box, Draw.DEFAULT_COLOR)
            
            caption = [f'{label[i]} {score[i]:.2f}%' for i in range(len(label))]
            colors = [Draw.colors.get(i, Draw.DEFAULT_COLOR) for i in label]
            if identity != '':
                colors += [Draw.DEFAULT_COLOR]
                caption += [identity]
            Draw.draw_text(det_img, caption, box, colors, font_size, text_thickness)

        return det_img

    @staticmethod
    def draw_box(image: np.ndarray, box: np.ndarray, color: Tuple[int, int, int] = DEFAULT_COLOR,
                thickness: int = 2) -> np.ndarray:
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    @staticmethod
    def draw_text(image: np.ndarray, caption: list, box: np.ndarray, colors: list,
                font_size: float = 0.001, text_thickness: int = 2) -> np.ndarray:
        x1, y1, x2, y2 = box.astype(int)

        for color, text in zip(colors, caption):
            (tw, th), _ = cv2.getTextSize(text=text, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=font_size, thickness=text_thickness)
            th = int(th * 1.2)

            cv2.rectangle(image, (x1, y1),
                        (x1 + tw, y1 - th), color, -1)

            cv2.putText(image, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), text_thickness, cv2.LINE_AA)

            y1 = y1 - th - 2

    @staticmethod
    def draw_masks(image: np.ndarray, boxes: np.ndarray, mask_alpha: float = 0.3) -> np.ndarray:
        mask_img = image.copy()

        # Draw bounding boxes and labels of detections
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)

            # Draw fill rectangle in mask image
            cv2.rectangle(mask_img, (x1, y1), (x2, y2), Draw.DEFAULT_COLOR, -1)

        return cv2.addWeighted(mask_img, mask_alpha, image, 1 - mask_alpha, 0)