import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import cv2


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([1,0,0, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def mask_to_bbox(mask, show_debug_message = False, threshold_min = 0.05, threshold_max = 0.25):
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x1 = mask.shape[1]
    y1 = mask.shape[0]
    x2 = 0
    y2 = 0

    for contour in contours:
        obj_x, obj_y, obj_w, obj_h = cv2.boundingRect(contour)
        if obj_x < x1:
            x1 = obj_x
        if obj_y < y1:
            y1 = obj_y
        if obj_x + obj_w > x2:
            x2 = obj_x + obj_w
        if obj_y + obj_h > y2:
            y2 = obj_y + obj_h
    
    ratio = ((x2-x1) * (y2-y1)) / (mask.shape[0] * mask.shape[1])
    if show_debug_message:
        print("ratio:", ratio)
    if ratio < threshold_min or ratio > threshold_max:
        return None
    return [x1, y1, x2, y2]