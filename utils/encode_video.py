import cv2
import torchvision.transforms as T
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import functional as F 
from utils.get_data_util import read_video_frames

# transform
input_mean = [0.48145466, 0.4578275, 0.40821073]
input_std = [0.26862954, 0.26130258, 0.27577711]

def get_index(num_frames, num_segments, centered=True):
    if centered:
        seg_size = float(num_frames - 1) / num_segments
        start = int(seg_size / 2)
    else:
        seg_size = float(num_frames - 1) / (num_segments - 1)
        start = 0
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets

def load_video(video_file, num_sampled_frames=8, return_msg=False, start_time=None, end_time=None,resoltion=224):
    sampled_frames, sampled_seconds, duration = read_video_frames(
        video_file, num_samples=num_sampled_frames, start=start_time, end=end_time, return_sampled_seconds=True
    )
    sampled_frames = sampled_frames.transpose(0,1)
    images = [image.permute(1,2,0).numpy().astype('uint8') for image in sampled_frames]
    if return_msg:
        sec = ", ".join([str(round(sec, 1)) for sec in sampled_seconds])
        msg = f"The video contains {sampled_seconds.shape[0]} frames sampled at {sec} seconds."
        return sampled_frames, msg, images
    else:
        return sampled_frames, images

def encode_video(sampled_frames, model, device1, device2):
    my_crop_transform = T.Compose([
        T.Resize((224,224), interpolation=InterpolationMode.BICUBIC),
        T.Lambda(lambda i : torch.clip(i, 0, 255)),
    ])
    my_value_transform = T.Compose([
        T.Lambda(lambda i : i /255),
        T.Normalize(mean=input_mean, std=input_std),
    ])
    resized_images = my_crop_transform(sampled_frames)
    torch_imgs = my_value_transform(resized_images)
    video_input = torch_imgs[None].to(torch.float16).to(device2)
    
    img_list = []
    with torch.no_grad():
        image_emb, _ = model.encode_img(video_input, "Watch the video and answer the question.")

    img_list.append(image_emb.to(device1))
    return img_list
    
def encode_video_segment(video_file, model, num_sampled_frames=8, return_msg=False, start_time=None, end_time=None, device1=None, device2=None):
    sampled_frames, msg, images = load_video(
        video_file, num_sampled_frames=num_sampled_frames, return_msg=True, start_time=start_time, end_time=end_time
    )
    img_list = encode_video(sampled_frames, model, device1, device2)
    return images, img_list

def load_frame(video_path, frame_id):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()
    path = "/home/ql/ours/examples/"
    img = Image.fromarray(frame)
    img.save(path+ f"frame_{frame_id}.png")
    return img
    
