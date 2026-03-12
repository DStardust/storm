import sys
# sys.path.append('/home/ql/ours/models/Ask-Anything/video_chat2')
import json
import os
import cv2
from PIL import Image
import numpy as np
import torch
from torchvision.transforms.functional import InterpolationMode
from decord import VideoReader, cpu
import torchvision.transforms as T
from dataset.video_transforms import (
    GroupNormalize, GroupScale, GroupCenterCrop, 
    Stack, ToTorchFormatTensor
)


# with open('/home/ql/dataset/ANet/train.json') as f:
#     act_train_content = json.load(f)
    
# with open('/home/ql/dataset/ANet/val_1.json') as f:
#     act_val_content = json.load(f)
    
# with open('/home/ql/dataset/ANet/val_2.json') as f:
#     act_test_content = json.load(f)

# act_train_anno = {}
# act_val_anno = {}
# act_test_anno = {}

# for video_name in act_train_content:
#     if video_name not in act_train_anno:
#         act_train_anno[video_name] = []
#     for ts, sent in zip(act_train_content[video_name]['timestamps'], act_train_content[video_name]['sentences']):
#         act_train_anno[video_name].append((ts[0], ts[1], sent))

# for video_name in act_val_content:
#     if video_name not in act_val_anno:
#         act_val_anno[video_name] = []
#     for ts, sent in zip(act_val_content[video_name]['timestamps'], act_val_content[video_name]['sentences']):
#         act_val_anno[video_name].append((ts[0], ts[1], sent))
        
# for video_name in act_test_content:
#     if video_name not in act_test_anno:
#         act_test_anno[video_name] = []
#     for ts, sent in zip(act_test_content[video_name]['timestamps'], act_test_content[video_name]['sentences']):
#         act_test_anno[video_name].append((ts[0], ts[1], sent))
        
# act_train_video_names = list(act_train_anno.keys())
# act_val_video_names = list(act_val_anno.keys())
# act_test_video_names = list(act_test_anno.keys())

# act_video_dirs = [
#     '/dataset/activitynet/videos/v1-2/train/',
#     '/dataset/activitynet/videos/v1-2/val/',
#     '/dataset/activitynet/videos/v1-2/test/',
#     '/dataset/activitynet/videos/v1-3/train_val/',
#     '/dataset/activitynet/videos/v1-3/test/',
# ]    

def act_get_file_path(video_name):
    video_path = None
    for video_dir in act_video_dirs:
        if os.path.exists(os.path.join(video_dir, video_name+'.mp4')):
            video_path = os.path.join(video_dir, video_name+'.mp4')
        elif os.path.exists(os.path.join(video_dir, video_name+'.mkv')):
            video_path = os.path.join(video_dir, video_name+'.mkv')
        if video_path is not None:
            break
    return video_path

def load_frame(video_path, frame_id):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()
    # path = "/home/ql/ours/examples/"
    img = Image.fromarray(frame)
    # img.save(path+ f"frame_{frame_id}.png")
    return img

def load_video(video_path, num_segments=8, return_msg=False, resolution=224, start_time = None, end_time = None):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    fps = float(vr.get_avg_fps())
    if start_time is None and end_time is None:
        num_frames = len(vr)
        frame_indices = get_index(num_frames, num_segments, 0)
    else:
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        if end_frame > len(vr):
            end_frame = len(vr)
        num_frames = end_frame - start_frame
        frame_indices = get_index(num_frames, num_segments, start_frame)

    # transform
    crop_size = resolution
    scale_size = resolution
    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]

    transform = T.Compose([
        GroupScale(int(scale_size), interpolation=InterpolationMode.BICUBIC),
        GroupCenterCrop(crop_size),
        Stack(),
        ToTorchFormatTensor(),
        GroupNormalize(input_mean, input_std) 
    ])

    images_group = list()
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].numpy())
        images_group.append(img)
    torch_imgs = transform(images_group)
    if return_msg:
        sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
        msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
        return torch_imgs, msg
    else:
        return torch_imgs

def get_index(num_frames, num_segments, start_frame):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    offsets += start_frame
    return offsets