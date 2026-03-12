import cv2
import numpy as np
import base64
import os

from io import BytesIO
from PIL import Image

def extract_keyframes_cv2(video_path, threshold=15.0, min_frame_gap=4):
    """
    Adaptive Keyframe Extraction optimized for Egocentric HOI Videos.
    - Uses structural difference (frame diff) rather than global histogram.
    - Captures small, sudden local movements (like hands interacting).
    - Ignores massive global shifts (like rapid head turning where 80%+ pixels completely change).
    - Forces a minimum gap `min_frame_gap` to prevent frame explosion during continuous action.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dimensions = [width, height]

    keyframes = []
    frame_count = 0
    frame_count_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total Frames: {frame_count_total}")

    prev_gray = None
    last_keyframe_idx = -min_frame_gap  # Allow frame 0 to be captured

    # Force a keyframe at specific time intervals regardless of motion (e.g., every 1.5 seconds)
    force_interval = int(fps * 1.5) if fps > 0 else 10

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Optional: focus on the center/bottom area where hands usually operate
        # cropped_frame = frame[int(height*0.2):, int(width*0.1):int(width*0.9)]
        
        # GaussianBlur is crucial to ignore video compression artifacts and tiny noise
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if prev_gray is None:
            prev_gray = gray
            keyframes.append(frame_count)
            last_keyframe_idx = frame_count
            frame_count += 1
            continue

        if frame_count - last_keyframe_idx >= force_interval:
            keyframes.append(frame_count)
            last_keyframe_idx = frame_count
            prev_gray = gray
            frame_count += 1
            continue

        if frame_count - last_keyframe_idx >= min_frame_gap:
            # Compute absolute difference between current and previous frame
            frame_delta = cv2.absdiff(prev_gray, gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            
            # Count the percentage of changed pixels
            changed_pixels = cv2.countNonZero(thresh)
            total_pixels = gray.shape[0] * gray.shape[1]
            change_ratio = (changed_pixels / float(total_pixels)) * 100.0

            # 1. change_ratio > threshold: Normal local movement (hand reaching in, objects moving)
            # 2. change_ratio < 70.0: Ignore massive full-screen blurring/whip pans common in ego-video
            if change_ratio > threshold and change_ratio < 70.0:
                keyframes.append(frame_count)
                last_keyframe_idx = frame_count
                prev_gray = gray

        frame_count += 1

    # Ensure the very last frame is always included for terminal state
    if frame_count_total - 1 not in keyframes and frame_count_total > 0:
        keyframes.append(frame_count_total - 1)

    cap.release()
    return keyframes, frame_count_total, fps, dimensions

def videoframes_to_base64(video_path, frame_list, save_frames = False, save_dir = None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    if save_frames and save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    base64_list = []
    for frame_index in frame_list:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if ret:
            _, buffer = cv2.imencode('.png', frame)
            base64_encoded = base64.b64encode(buffer).decode('utf-8')
            base64_list.append(base64_encoded)
            if save_frames:
                file_name = f"{save_dir}/{frame_index}.jpg"
                try:
                    with open(file_name, 'wb') as f:
                        f.write(buffer)
                except Exception as e:
                    print(f"Error saving frame {frame_index}: {e}")
        else:
            base64_list.append(None)
            print("Invalid Index:", frame_index)
    cap.release()
    return base64_list

def cv2_to_base64(cv2_data):
    _, buffer = cv2.imencode('.png', cv2_data)
    base64_encoded = base64.b64encode(buffer).decode('utf-8')
    return base64_encoded
    

def base64_to_PIL(base64_data):
    raw_data = base64.b64decode(base64_data)
    raw_bytes = BytesIO(raw_data)
    PIL_image = Image.open(raw_bytes)
    return PIL_image

def frame_add_bbox(frame_PIL, bbox):
    frame_array = np.array(frame_PIL)
    if len(frame_array.shape) == 3 and frame_array.shape[2] == 3:
        frame_cv2 = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
    else:
        frame_cv2 = frame_array

    cv2.rectangle(frame_cv2, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 3)
    return frame_cv2

def frame_add_bbox_b64(frame_b64, bbox):
    """
     Return CV2-style frame with bounding box
    """
    frame_PIL = base64_to_PIL(frame_b64)
    frame_array = np.array(frame_PIL)
    if len(frame_array.shape) == 3 and frame_array.shape[2] == 3:
        frame_cv2 = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
    else:
        frame_cv2 = frame_array

    cv2.rectangle(frame_cv2, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 3)
    return frame_cv2

def frame_add_bboxes_and_labels_b64(frame_b64, obj_bbox_dict, save_frames = False, save_name = None):

    frame_PIL = base64_to_PIL(frame_b64)
    frame_array = np.array(frame_PIL)
    if len(frame_array.shape) == 3 and frame_array.shape[2] == 3:
        frame_cv2 = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
    else:
        frame_cv2 = frame_array

    for i in range(len(obj_bbox_dict["labels"])):
        bbox = obj_bbox_dict["bboxes"][i]
        cv2.rectangle(frame_cv2, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)

    for i in range(len(obj_bbox_dict["labels"])):
        bbox = obj_bbox_dict["bboxes"][i]
        label = obj_bbox_dict["labels"][i] + '#' + str(obj_bbox_dict["reference"][i])
        label_position = ((bbox[0]+bbox[2])//2, (bbox[1]+bbox[3])//2)
        cv2.putText(frame_cv2, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255) , 2)

    if save_frames:
        save_dir = os.path.dirname(save_name)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(save_name, frame_cv2)
    
    return frame_cv2

def videoframes_to_files(video_path, frame_list, save_dir):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    os.makedirs(save_dir, exist_ok=True)
    for frame_index in frame_list:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if ret:
            _, buffer = cv2.imencode('.png', frame)
            file_name = f"{save_dir}/{frame_index}.jpg"
            try:
                with open(file_name, 'wb') as f:
                    f.write(buffer)
            except Exception as e:
                print(f"Error saving frame {frame_index}: {e}")
        else:
            print("Invalid Index:", frame_index)
    cap.release()
    return True