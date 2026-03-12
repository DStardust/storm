import pandas as pd
import json
import numpy as np
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.data.encoded_video_pyav import thwc_to_cthw, _pyav_decode_stream
import torch
import os, math
import cv2

def get_index_frame(video, indices):
    frame_idxs = indices

    try:
        outputs = video._av_reader.get_batch(frame_idxs)
    except Exception as e:
        print(f"Failed to decode video with Decord: {video._video_name}. {e}")
        raise e

    video = outputs

    if video is not None:
        if not isinstance(video, torch.Tensor):
            video = torch.from_numpy(video.asnumpy())
        video = video.to(torch.float32)
        video = thwc_to_cthw(video)

    return video

def get_sec_frame(video, seconds):
    if isinstance(seconds, float):
        indices = [min(round(video._fps * seconds), len(video._av_reader)-1)]
    else:
        indices = [min(round(video._fps * sec), len(video._av_reader)-1) for sec in seconds]
    frame_idxs = indices

    try:
        outputs = video._av_reader.get_batch(frame_idxs)
    except Exception as e:
        print(f"Failed to decode video with Decord: {video._video_name}. {e}")
        raise e

    video = outputs

    if video is not None:
        if not isinstance(video, torch.Tensor):
            video = torch.from_numpy(video.asnumpy())
        video = video.to(torch.float32)
        video = thwc_to_cthw(video)

    return video

def get_clip_audio(video, start_sec, end_sec, target_sample_rate=16000):
    sample_rate = video._av_reader._AVReader__audio_reader.sample_rate
    num_audio_signals = video._av_reader._AVReader__audio_reader.shape[1]
    
    start_idx = int(sample_rate * start_sec)
    end_idx = math.ceil(sample_rate * end_sec)
    start_idx, end_idx = min(start_idx, num_audio_signals-1), min(end_idx, num_audio_signals-1)
    
    frame_idxs = np.linspace(start_idx, end_idx, int(target_sample_rate*(end_sec - start_sec))).astype('int64')
    
    audio_arr = torch.Tensor(video._av_reader._AVReader__audio_reader._array[0, frame_idxs])
    audio = audio_arr.to(torch.float32)
    return audio

def read_video_file(video_path, decoder='decord', decode_audio=False):
    video = EncodedVideo.from_path(video_path, decoder=decoder, decode_audio=decode_audio)
    return video

def read_video_frames(video, sampled_seconds=None, num_samples=None, start=None, end=None, centered=True, decode_audio=False, return_sampled_seconds=False):
    duration = video.duration
    if sampled_seconds is None:
        if start is None:
            start = 0
        if end is None:
            end = video.duration
        duration = end - start
        if num_samples is None:
            num_samples = round(duration)
        if centered:
            clip_len = duration / num_samples
            sampled_seconds = np.arange(num_samples) * clip_len + 0.5 * clip_len + start
        else:
            clip_len = duration / (num_samples-1)
            sampled_seconds = np.arange(num_samples) * clip_len + start
    video_data = get_sec_frame(video, sampled_seconds)
    if return_sampled_seconds:
        return video_data, sampled_seconds, duration
    else:
        return video_data, duration

def scaling_video(raw_video_path, processed_video_path, fps = 4, max_token_lenth = 51200, token_pixels = 32 * 32):

    os.makedirs(processed_video_path, exist_ok=True)

    video_name = os.path.basename(raw_video_path)
    video_root, video_ext = os.path.splitext(video_name)
            
    # 打开视频文件
    cap = cv2.VideoCapture(raw_video_path)

        
    if not cap.isOpened():
        print(f"Failed to open '{raw_video_path}'")
        return None
            
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        raw_fps = cap.get(cv2.CAP_PROP_FPS)        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / raw_fps if raw_fps > 0 else 0

        target_frame_count = int(duration * fps)

        if target_frame_count <= 0:
            return None

        # Qwen2.5-VL 常用 28x28 像素约等于 1 个视觉 token。
        # Qwen3/Qwen3.5 视觉模型常用 32x32 像素约等于 1 个视觉 token。
        total_token_lenth = math.ceil(width * height / token_pixels) * target_frame_count

        print(f"Raw_Token_Count: {total_token_lenth}" )

        if total_token_lenth <= max_token_lenth:

            scaled_video_name = f"{video_root}_scaled_{width}x{height}_{fps}fps{video_ext}"
            scaled_video_path = os.path.join(processed_video_path, scaled_video_name)

            if os.path.exists(scaled_video_path):
                return scaled_video_path

            # 获取视频编码格式
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            # 创建视频写入对象
            out = cv2.VideoWriter(
                scaled_video_path,
                fourcc,
                fps,
                (width, height)
            )
            
            for i in range(target_frame_count):
                
                target_time = i / fps
                # 计算对应的原始帧索引
                raw_frame_idx = int(round(target_time * raw_fps))
                             
                # 确保索引在有效范围内并定位帧
                if raw_frame_idx >= frame_count:
                    raw_frame_idx = frame_count - 1                
                cap.set(cv2.CAP_PROP_POS_FRAMES, raw_frame_idx)
                success, frame = cap.read()
                
                if success:
                    out.write(frame)
                
            return scaled_video_path

        else:
            # 计算所需的缩放比例
            current_pixels_per_frame = width * height
            max_pixels_per_frame = (max_token_lenth * token_pixels) / target_frame_count
            scale_ratio = math.sqrt(max_pixels_per_frame / current_pixels_per_frame)
            
            # 计算新的分辨率，保持宽高比
            new_width = max(1, int(width * scale_ratio) )
            new_height = max(1, int(height * scale_ratio) )
                      
            # 构建输出路径
            scaled_video_name = f"{video_root}_scaled_{new_width}x{new_height}_{fps}fps{video_ext}"
            scaled_video_path = os.path.join(processed_video_path, scaled_video_name)

            if os.path.exists(scaled_video_path):
                return scaled_video_path
            
            # 根据文件扩展名选择合适的编码器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            # 创建视频写入对象
            out = cv2.VideoWriter(
                scaled_video_path,
                fourcc,
                fps,
                (new_width, new_height)
            ) 
            # 处理并写入视频帧
            for i in range(target_frame_count):
                target_time = i / fps
                raw_frame_idx = int(round(target_time * raw_fps))
                
                # 确保索引在有效范围内并定位帧
                if raw_frame_idx >= frame_count:
                    raw_frame_idx = frame_count - 1                
                cap.set(cv2.CAP_PROP_POS_FRAMES, raw_frame_idx)
                success, frame = cap.read() 

                if success:
                    resized_frame = cv2.resize(frame, (new_width, new_height))
                    out.write(resized_frame)
            
            out.release() # 释放资源
            return scaled_video_path
                
    except Exception as e:
        print(f"Failure: {str(e)}")
        return None
    finally:
        cap.release()