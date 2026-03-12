import argparse
import os
import pickle
import re
import tqdm
import copy
import torch
import json

from io import BytesIO
from utils.get_data_util import *
# from utils.load_data import *
from utils.keyframe_utils import *
from utils.api_utils import *
from utils.sam2_utils import *
from scenedetect import detect, ContentDetector
from transformers import AutoProcessor, AutoModelForCausalLM
from sam2.build_sam import build_sam2_video_predictor
from PIL import Image
from torchvision.ops import box_iou

import utils.stsg as stsg
import numpy as np

# from GroundingDino import *
# from prompt import *

import subprocess
import random
import cv2

HUMAN_ENTITY_LABELS = {
    "person", "people", "human", "man", "woman", "boy", "girl",
    "lady", "gentleman", "adult", "child"
}

EGO_OPERATOR_BODY_LABELS = [
    "hands",
    "left hand",
    "right hand",
    "feet",
    "left foot",
    "right foot",
    "left wrist",
    "right wrist",
    "left arm",
    "right arm",
    "torso",
    "operator body"
]

NON_OPERATOR_HUMAN_LABELS = [
    "other person",
    "non-operator human",
    "not sure"
]

OPERATOR_PART_MAX_SAMPLES = 12
OPERATOR_PART_MIN_SAMPLES = 4

def iou(mask1, mask2):
    tensor1 = torch.from_numpy(mask1)
    tensor2 = torch.from_numpy(mask2)
    intersection = tensor1 * tensor2
    union = torch.clamp(tensor1 + tensor2, 0, 1)

    intersection_area = torch.sum(intersection)
    union_area = torch.sum(union)

    iou = intersection_area / union_area

    return iou

def sample_object_frame_numbers(obj, frames_for_sam2, max_samples=OPERATOR_PART_MAX_SAMPLES):
    visible_frames = []
    for frame_idx, frame_no in enumerate(frames_for_sam2):
        if frame_idx in obj["mask"] and np.sum(obj["mask"][frame_idx]) > 0:
            visible_frames.append(frame_no)

    if not visible_frames:
        return []

    if len(visible_frames) <= max_samples:
        return visible_frames

    sampled_indices = np.linspace(0, len(visible_frames) - 1, num=max_samples, dtype=int)
    return [visible_frames[idx] for idx in sampled_indices]

def pad_base64_frame_list(base64_frame_list, min_frames=4):
    valid_frames = [frame for frame in base64_frame_list if frame]
    if not valid_frames:
        return []

    while len(valid_frames) < min_frames:
        valid_frames.append(valid_frames[-1])

    return valid_frames

def save_operator_part_debug_frame(debug_dir, object_index, frame_no, frame_cv2):
    os.makedirs(debug_dir, exist_ok=True)
    debug_path = os.path.join(debug_dir, f"object_{object_index}_frame_{frame_no}.jpg")
    cv2.imwrite(debug_path, frame_cv2)
    return debug_path

def normalize_operator_body_label(raw_response):
    response = raw_response.lower().strip()

    ordered_candidates = sorted(
        EGO_OPERATOR_BODY_LABELS + NON_OPERATOR_HUMAN_LABELS,
        key=len,
        reverse=True,
    )
    for candidate in ordered_candidates:
        if candidate in response:
            return candidate

    alias_map = {
        "both hands": "hands",
        "two hands": "hands",
        "left and right hand": "hands",
        "right and left hand": "hands",
        "hand": "hands",
        "left palm": "left hand",
        "right palm": "right hand",
        "both feet": "feet",
        "two feet": "feet",
        "left and right foot": "feet",
        "right and left foot": "feet",
        "foot": "feet",
        "left forearm": "left arm",
        "right forearm": "right arm",
        "left leg": "left foot",
        "right leg": "right foot",
        "body": "operator body",
        "camera wearer": "operator body",
        "wearer": "operator body",
    }
    for alias, normalized_label in alias_map.items():
        if alias in response:
            return normalized_label

    return None

def relabel_egocentric_operator_parts(video_path, frames_for_sam2, all_object_list, video_od_list, debug_dir):
    for obj in all_object_list:
        if obj["label"] not in HUMAN_ENTITY_LABELS:
            continue

        sampled_frame_numbers = sample_object_frame_numbers(obj, frames_for_sam2)
        if not sampled_frame_numbers:
            continue

        if DEBUG_MODE:
            print(
                f"Operator relabel object {obj['index']} ({obj['label']}) sampled frames: {sampled_frame_numbers}"
            )

        sampled_frames_b64 = videoframes_to_base64(video_path, sampled_frame_numbers)
        highlighted_frames_b64 = []
        debug_image_paths = []

        for frame_no, frame_b64 in zip(sampled_frame_numbers, sampled_frames_b64):
            frame_idx = frames_for_sam2.index(frame_no)
            obj_mask = obj["mask"].get(frame_idx)
            if obj_mask is None or np.sum(obj_mask) == 0:
                continue

            obj_bbox = mask_to_bbox(obj_mask, threshold_min=0.05, threshold_max=0.25)
            if not obj_bbox:
                continue

            highlighted_frame_cv2 = frame_add_bbox_b64(frame_b64, obj_bbox)
            highlighted_frames_b64.append(cv2_to_base64(highlighted_frame_cv2))
            debug_image_paths.append(
                save_operator_part_debug_frame(debug_dir, obj["index"], frame_no, highlighted_frame_cv2)
            )

        if not highlighted_frames_b64:
            continue

        highlighted_frames_b64 = pad_base64_frame_list(
            highlighted_frames_b64,
            min_frames=OPERATOR_PART_MIN_SAMPLES,
        )
        if not highlighted_frames_b64:
            continue

        if DEBUG_MODE:
            print(
                f"Operator relabel object {obj['index']} sending {len(highlighted_frames_b64)} frames to Qwen"
            )

        prompt = f"""In a first-person egocentric video, the red boxed region across these frames is the same tracked entity.
Determine whether it is a body part of the camera wearer / operator.
If it is the operator's body part, answer with exactly one label from: {', '.join(EGO_OPERATOR_BODY_LABELS)}.
If it is not the operator's body part, answer with exactly one label from: {', '.join(NON_OPERATOR_HUMAN_LABELS)}.
Return only the label, with no explanation."""
        response = api_response_base64piclist(
            client=qwen_client,
            model=Model_dic["QWEN-VL-235B"],
            prompt=prompt,
            base64_picture_list=highlighted_frames_b64,
        )

        normalized_label = normalize_operator_body_label(response)
        if normalized_label is None:
            print(f"Operator part relabel undecided for object {obj['index']}: {response}")
            continue

        obj["original_label"] = obj["label"]
        obj["operator_part_response"] = response
        obj["is_operator_part"] = normalized_label in EGO_OPERATOR_BODY_LABELS
        obj["operator_part_debug_images"] = debug_image_paths

        if not obj["is_operator_part"]:
            continue

        obj["label"] = normalized_label
        print(f"Relabeled human object {obj['index']} from {obj['original_label']} to {normalized_label}")

        for od_entry in video_od_list:
            for label_idx, reference in enumerate(od_entry["OD"]["reference"]):
                if reference == obj["index"]:
                    od_entry["OD"]["labels"][label_idx] = normalized_label

if os.getenv("API_QWEN_OMNI"):
    qwen_client = create_client(os.getenv("API_QWEN_OMNI"), "https://dashscope.aliyuncs.com/compatible-mode/v1")
else:
    print("API Client Creation Failed")

DEBUG_MODE = True

Model_dic = {"QWEN-OMNI": "qwen-omni-turbo-latest", 
             "QWEN-PLUS":"qwen-plus-latest",
             "QWEN-VL-MAX": "qwen3-vl-flash",
             "QWEN-VL-235B": "qwen3-vl-235b-a22b-instruct"}

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

OD_model_path = "/mnt/public/daiyang/benchmark_vstorm/models/Florence2-large"
OD_model = AutoModelForCausalLM.from_pretrained(OD_model_path, torch_dtype=torch_dtype, trust_remote_code=True).to(device)
OD_processor = AutoProcessor.from_pretrained(OD_model_path, trust_remote_code=True)

SAM2_checkpoint = "/mnt/public/daiyang/benchmark_vstorm/models/sam2/checkpoints/sam2.1_hiera_large.pt"
SAM2_model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_predictor = build_sam2_video_predictor(SAM2_model_cfg, SAM2_checkpoint, device=device)

# 注意修改
write_path = "../processed_stsg/_test_0311.json"
os.makedirs(os.path.dirname(write_path), exist_ok=True)

video_path_list = [
    "/mnt/public/daiyang/benchmark_vstorm/stsg/raw_videos/fine-ego/selected_final/P01_102_1.mp4"
]
# 新增的视频输出path(文件夹路径)
processed_video_path = "/mnt/public/daiyang/benchmark_vstorm/stsg/processed_video"
operator_part_debug_root = "../debug_operator_parts"
example_frames_root = "../example_frames"

# 基于video创造视频级别STSG
count = 0
for video_id in range(0, len(video_path_list)):
    # 视频处理
    raw_video_path = video_path_list[video_id]
    # 视频缩放
    video_path = scaling_video(raw_video_path, processed_video_path, fps = 4, max_token_lenth= 51200)

    segment_list = detect(video_path, ContentDetector(threshold=21))
    segment_moments = [[s.get_seconds(), e.get_seconds()] for s, e in segment_list] # 获得时间起点与终点
    video_name = (video_path.split('/')[-1]).split('.')[0]
    saving_dir = f"../filtered_frames/{video_name}"
    operator_part_debug_dir = os.path.join(operator_part_debug_root, video_name)
    example_frames_dir = os.path.join(example_frames_root, video_name)
    sgs = []

    if DEBUG_MODE:
        print("saving_dir:", saving_dir)
        print("Segment_list:", segment_list)

    if len(segment_list) == 0:
        print(f"{video_name} have no segment, parsing failed")
        continue

    for i, s in enumerate(segment_moments):
        print('Scene {}: {:.1f}s, {:.1f}s'.format(i, *s))

    keyframes, total_frames, fps, dim = extract_keyframes_cv2(video_path)
    print(keyframes, total_frames, fps)
    frames_for_sam2 = []
    frames_for_sam2_raw = []
    video_keyframes = []
    upsampling_base64_segments = []

    # 关键帧的划分
    for segment_id in range(len(segment_list)):

        # 获得片段起止时间戳和帧编号
        seg_start_time, seg_end_time = segment_list[segment_id][0].get_seconds(), segment_list[segment_id][1].get_seconds()
        seg_start_frame, seg_end_frame = segment_list[segment_id][0].get_frames(), segment_list[segment_id][1].get_frames()
        if seg_end_frame >= total_frames:
            seg_end_frame = total_frames - 1
        
        # if seg_end_time-seg_start_time > 30:
        #     break

        filtered_keyframes = []
        filtered_keyframes.append(seg_start_frame)
        for frame in keyframes:
            if frame > seg_start_frame and frame < seg_end_frame:
                filtered_keyframes.append(frame)
        filtered_keyframes.append(seg_end_frame)
        print("filtered_keyframes:", filtered_keyframes)

        upsampling_keyframes = None
        if len(filtered_keyframes) < 4:
            if len(filtered_keyframes) == 2:
                interval = (filtered_keyframes[1]-filtered_keyframes[0])//3
                upsampling_keyframes = [filtered_keyframes[0], 
                                          filtered_keyframes[0] + 1*interval, 
                                          filtered_keyframes[0] + 2*interval,
                                          filtered_keyframes[1]]
            elif len(filtered_keyframes) == 3:
                upsampling_keyframes = [filtered_keyframes[0],
                                          (filtered_keyframes[0]+filtered_keyframes[1])//2,
                                          filtered_keyframes[1],
                                          (filtered_keyframes[1]+filtered_keyframes[2])//2,
                                          filtered_keyframes[2]]
            else:
                continue
        # elif len(filtered_keyframes) > 6:
        #     print("Too many keyframes! Select first 6 keyframes")
        #     filtered_keyframes = filtered_keyframes[0:6]
        #     upsampling_keyframes = filtered_keyframes
        else:
            upsampling_keyframes = filtered_keyframes

        upsampling_flist_base64 = videoframes_to_base64(video_path, upsampling_keyframes)
        upsampling_base64_segments.append(upsampling_flist_base64)
        
        video_keyframes.append(filtered_keyframes)
        frames_for_sam2_raw.extend(filtered_keyframes)

    frames_for_sam2 = list(set(frames_for_sam2_raw))
    frames_for_sam2.sort()

    if DEBUG_MODE:
        print("frames_for_sam2:", frames_for_sam2)
    if len(frames_for_sam2) > 50:
        print(f"{video_name} has too many keyframes! Early-stopping to prevent massive cost!")
        continue

    os.makedirs(saving_dir, exist_ok=True)
    videoframes_to_files(video_path, frames_for_sam2, saving_dir)
    SAM2_inference_state = SAM2_predictor.init_state(video_path = saving_dir)
    
    # Object Detection
    all_object_list = []
    video_od_list = []

    for frame_id, frame_no in enumerate(frames_for_sam2):         
        frame_base64 = videoframes_to_base64(video_path, [frame_no])[0]

        # step 2.i.x: video_frames[frame_no]* ==OD==> frame_od_list
        frame_PIL = base64_to_PIL(frame_base64)
        OD_inputs = OD_processor(text="<OD>", images=frame_PIL, return_tensors="pt").to(device, torch_dtype)

        OD_generated_ids = OD_model.generate(
            input_ids=OD_inputs["input_ids"],
            pixel_values=OD_inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3)
        OD_generated_text = OD_processor.batch_decode(OD_generated_ids, skip_special_tokens=False)[0]
        OD_parsed_answer = OD_processor.post_process_generation(OD_generated_text, task="<OD>", image_size=(frame_PIL.width, frame_PIL.height))
        frame_od_list = OD_parsed_answer['<OD>']
        frame_od_list["reference"] = []

        frame_category_dict = {}

        # frame_od_list= {"bboxes": [box0, box1, ...], "labels": [label0, label1, ...]}
        for i in range(len(frame_od_list["bboxes"])):
            old_bbox = frame_od_list["bboxes"][i]
            new_bbox = [int(num) for num in old_bbox]
            frame_od_list["bboxes"][i]= new_bbox
            frame_od_list["reference"].append(-1)

        object_cite_list = [-1] * len(frame_od_list["labels"])

        # 根据sam2判断元素是否之前已经出现
        for idx in range(0, len(frame_od_list["labels"])):
            bbox = frame_od_list["bboxes"][idx]
            label = frame_od_list["labels"][idx].lower().strip()
            object_id = len(all_object_list)
            SAM2_predictor.reset_state(SAM2_inference_state)
            _, out_obj_ids, out_mask_logits = SAM2_predictor.add_new_points_or_box(
                inference_state=SAM2_inference_state,
                frame_idx=frames_for_sam2.index(frame_no),
                obj_id=object_id,
                box=bbox,
            )
            new_mask_bool = (out_mask_logits[0] > 0.0).cpu().numpy()
            new_mask = new_mask_bool[0].astype(int)
            # 判遍历已有的元素
            for exist_obj in all_object_list:
                if np.sum( exist_obj["mask"][frames_for_sam2.index(frame_no)])  > 0: # 物品在这一帧有出现
                    if iou(new_mask, exist_obj["mask"][frames_for_sam2.index(frame_no)]) > 0.9: #是同一物品
                        object_cite_list[idx] = exist_obj["index"]
                        frame_od_list["reference"][idx] = exist_obj["index"]
                        exist_obj["label"].append(label)
                        # print("Exist!")
                        break
            # 未找到元素，添加新的元素            
            if object_cite_list[idx] == -1:

                prompt=f"Given a picture, answer whether there is a {label} in this picture in a single word. You must select an answer between 'Yes' or 'No'. "
                response = api_response_base64picture(qwen_client, "qwen-vl-max", prompt, frame_base64)
                print("label: ", label, " response: ", response)

                if 'no' in response.lower():
                    frame_cv2_with_box = frame_add_bbox_b64(frame_base64, bbox)
                    frame_b64_with_box = cv2_to_base64(frame_cv2_with_box)

                    prompt = f"Given a picture with bounding box, answer what kind of object is in the bounding box. Your answer must be in a single word."
                    response = api_response_base64picture(qwen_client, "qwen-vl-max", prompt, frame_b64_with_box)

                    label = response.lower().strip()
                    print("Recognition Failed! New label: ", label)

                sam2_video_segments = {}  # sam2_video_segments contains the per-frame segmentation results
                sam2_output_masks = {}
                for out_frame_idx, out_obj_ids, out_mask_logits in SAM2_predictor.propagate_in_video(SAM2_inference_state):
                    sam2_video_segments[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }
                    sam2_output_masks[out_frame_idx] = sam2_video_segments[out_frame_idx][object_id][0].astype(int)
                new_dict = {}
                new_dict["label"] = [label]
                new_dict["index"] = object_id
                new_dict["mask"] = sam2_output_masks
                new_dict["video_segment"] = sam2_video_segments
                new_dict["start_frame"] = frame_no
                new_dict["attributes"] = []
                all_object_list.append(new_dict)

                object_cite_list[idx] = object_id
                frame_od_list["reference"][idx] = object_id

        new_dict = {"frame": frame_no, 
                    "OD": frame_od_list}
        video_od_list.append(new_dict)

    for obj in all_object_list:
        label_calculate_table = {}
        for label in obj["label"]:
            if label in label_calculate_table:
                label_calculate_table[label] += 1
            else:
                label_calculate_table[label] = 1

        max_value = -1
        max_key = None
        for key in label_calculate_table:
            if label_calculate_table[key] > max_value:
                max_value = label_calculate_table[key]
                max_key = key

        # 投票失败
        if (list(label_calculate_table.values()).count(max_value)) > 1:

            base64_image_list = []

            for out_frame_idx in range(frames_for_sam2.index(obj["start_frame"]), len(frames_for_sam2)):
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                img = Image.open(os.path.join(saving_dir, f"{frame_no}.jpg"))

                axes[0].imshow(img)
                axes[0].set_title(f"frame {out_frame_idx} (no mask)")

                axes[1].imshow(img)
                axes[1].set_title(f"frame {out_frame_idx} (with mask)")
                for out_obj_id, out_mask in obj["video_segment"][out_frame_idx].items():
                    show_mask(out_mask, axes[1], obj_id=out_obj_id)
                        
                buffer = BytesIO()
                plt.savefig(buffer, format='jpg')
                buffer.seek(0)

                base64_image = base64.b64encode(buffer.read()).decode('utf-8')
                base64_image_list.append(base64_image)
                plt.close()
                    
            if len(base64_image_list) < 4:
                for i in range(len(base64_image_list), 4):
                    base64_image_list.append(base64_image_list[-1])
                    
            prompt=f"Given the object highlighted in red in the video, answer what is it in a single word. You must select an answer among {list(label_calculate_table.keys())}. "
            response = api_response_base64piclist(qwen_client, "qwen-vl-max", prompt, base64_image_list)
            print("Prompt:", prompt, "Response:", response)
            for label_ in label_calculate_table.keys():
                obj["label"] = label_
                if label_ in response.lower():
                    break
        else:
            obj["label"] = max_key

    print(video_od_list)

    # 避免Florence2遗漏物品
    for obj in all_object_list:
        for frame_id, frame_no in enumerate(frames_for_sam2):
            if frame_no >= obj["start_frame"]:
                if np.sum(obj["mask"][frames_for_sam2.index(frame_no)]) > 0:
                    if obj["index"] not in video_od_list[frame_id]["OD"]["reference"]:
                        obj_bbox = mask_to_bbox(obj["mask"][frames_for_sam2.index(frame_no)], threshold_min = 0.05, threshold_max = 0.25)

                        if obj_bbox:
                            bbox1 = torch.tensor(obj_bbox).unsqueeze(0)
                            obj_overlap_flag = 0
                            # 避免Sam2重复添加物品
                            for exist_obj_id, exist_obj_bbox in enumerate(video_od_list[frame_id]["OD"]["bboxes"]):
                                if video_od_list[frame_id]["OD"]["labels"][exist_obj_id] == obj["label"]: # 是同一类物品
                                    bbox2 = torch.tensor(exist_obj_bbox).unsqueeze(0)
                                    if box_iou(bbox1, bbox2) > 0.5:
                                        obj_overlap_flag = 1
                                        print("Sam2 Overlapping On ", obj["label"])
                                        break
                            if obj_overlap_flag == 0:
                                video_od_list[frame_id]["OD"]["bboxes"].append(obj_bbox)
                                video_od_list[frame_id]["OD"]["labels"].append(obj["label"])
                                video_od_list[frame_id]["OD"]["reference"].append(obj["index"])

    print(video_od_list)

    relabel_egocentric_operator_parts(
        video_path=video_path,
        frames_for_sam2=frames_for_sam2,
        all_object_list=all_object_list,
        video_od_list=video_od_list,
        debug_dir=operator_part_debug_dir,
    )

    if DEBUG_MODE:
        print("video_od_list after operator relabel:", video_od_list)
    
    if len(all_object_list) > 50:
        print(f"{video_name} has {len(all_object_list)} elements! Early-stopping to prevent massive cost.")
        continue

    # for segment_id in range(2, len(segment_list)):
    for segment_id in range(len(segment_list)):

        seg_start_time, seg_end_time = segment_list[segment_id][0].get_seconds(), segment_list[segment_id][1].get_seconds()
        seg_start_frame, seg_end_frame = segment_list[segment_id][0].get_frames(), segment_list[segment_id][1].get_frames()
        if seg_end_frame >= total_frames:
            seg_end_frame = total_frames - 1
    
        filtered_keyframes = video_keyframes[segment_id]

        upsampling_flist_base64 = upsampling_base64_segments[segment_id]

        seg_response = api_response_base64piclist(qwen_client, "qwen-vl-max", 
                                                  "What happend in this video segment?", 
                                                  upsampling_flist_base64)
        event = stsg.Event(
            segment_id= segment_id,
            event_description = seg_response.replace(".", ""),
            start_time=seg_start_time,
            end_time=seg_end_time
        )
        segment_graph = stsg.SegmentSTSG(segment_id, seg_start_time, seg_end_time, event, filtered_keyframes)
        
        for frame_id, frame_no in enumerate(filtered_keyframes): 

            frame_base64 = videoframes_to_base64(video_path, [frame_no], save_frames = True, save_dir = example_frames_dir)[0]
            frame_PIL = base64_to_PIL(frame_base64)

            frame_graph = stsg.FrameSG(frame_no, frame_no/fps)
            for od_list in video_od_list:
                if od_list["frame"] == frame_no:
                    frame_od_list = od_list["OD"]
                    break
            

            # Step 2.ii
            for od_index in range(0, len(frame_od_list["labels"])):
                # Step 2.ii.a 以NL的形式生成帧中各元素的属性
                obj_bbox = frame_od_list["bboxes"][od_index]
                obj_reference = frame_od_list["reference"][od_index]
                obj_label = all_object_list[obj_reference]["label"]

                frame_with_bbox = cv2_to_base64(frame_add_bbox(frame_PIL=frame_PIL, bbox=obj_bbox))
                prompt = f"Given the object: '{obj_label}' highlighted in the box, \
describe its attributes in the image that primarily include adjectives or present participles (-ing forms). \
Avoid any irrelevant details or complex explanations."
                obj_NL_attribute = api_response_base64picture(client= qwen_client, model = Model_dic["QWEN-VL-MAX"], 
                                                              prompt= prompt, base64_picture = frame_with_bbox)
                # print("Obj_NL_attribute:", obj_NL_attribute)

                # Step 2.ii.b 将NL形式的元素属性转化为字典格式, eg.["object":..., "attributes":[...]]
                prompt = f"""For the given sentence, the task is to extract meaningful object label and attributes for the given object. Each attribute should include its type and value.
Let's take a few examples to understand how to extract.
Question: Given the sentence "A woman with long blonde hair is dancing on a park bench." Extract meaningful object label and attributes for the object "woman"
Answer: ["object":"woman", "attributes":["type":"hair", "value":"long blonde"; "type":"status", "value":"dancing"]
Question: Given the sentence "A black cat is sitting on a windowsill." Extract meaningful object label and attributes for the object "black cat"
Answer: ["object":"cat", "attributes":["type":"color", "value":"black"; "type":"status", "value":"sitting"]]
Question: Given the sentence "The blue car is parked." Extract meaningful object label and attributes for the object "blue car"
Answer: ["object":"car", "attributes":["type":"color", "value":"blue"; "type":"status", "value":"parked"]]
Please answer the following question.
"""
                full_prompt = prompt + f"Given the sentence \"{obj_NL_attribute}\", extract meaningful object label and attributes for the object \"{obj_label}\". Your answer must strictly follow the style of example, not json."
                response = api_response_textonly(client=qwen_client, model=Model_dic["QWEN-PLUS"], prompt=full_prompt)

                response = parsing_str(response)
                # print("Filtered_Obj_Attribute_Response:", response)
                
                # Parsing Label
                label_match = re.search(r'"object":"(.*?)"', response)                
                # Parsing Attribute
                attributes_matches = re.findall(r'"type":"(.*?)","value":"(.*?)"', response)
                
                # print("Step 2.ii.b attributes_matches:", attributes_matches)

                if label_match:
                    label = label_match.group(1)
                    Obj = stsg.Object(
                        segment_id = segment_id,
                        frame_id = frame_id,
                        object_id = len(frame_graph.objects),
                        object_name = obj_label.lower(),   # Test, Maybe wrong 
                        distinct_label = obj_reference,
                        time = frame_graph.time,
                        bbox = obj_bbox 
                    )
                    for match in attributes_matches:
                        attribute_type, attribute_value = match
                        attribute_value = attribute_value.replace('"', '').replace('[', '').replace(']', '').lower().strip()
                        Obj.add_attribute(stsg.Attribute(attribute_type, attribute_value))

                        new_attribute_dict = {"time": frame_graph.time, "attribute_type": attribute_type, "attribute_value": attribute_value}
                        all_object_list[obj_reference]["attributes"].append(new_attribute_dict)

                else:
                    # print("No matches found")
                    Obj = stsg.Object(
                        segment_id = segment_id,
                        frame_id = frame_id,
                        object_id = len(frame_graph.objects),
                        object_name = obj_label.lower(),
                        distinct_label = obj_reference,
                        time = frame_graph.time,
                        bbox = obj_bbox                      
                    )
                

                frame_graph.add_object(Obj)
                # print("Label:", obj_reference, "Object_In_Graph:", Obj)

                # Step 2.iii
            for obj in frame_graph.objects:
                obj_label = obj.object_name
                obj_distinct_label = obj.distinct_label
                obj_labels = [obj.object_name for obj in frame_graph.objects]

                # print("obj_labels:", obj_labels)

                obj_bbox = obj.bbox[-1]
                prompt = f"For the given object: '{obj}' highlighted in the box, \
describe all spatial or contact relationship between {obj_label} and other objects including {obj_labels} in the image.\
Ensure the subject of every sentence is {obj_label}, and do not include detailed descriptions of other objects."
                frame_with_bbox = cv2_to_base64(frame_add_bbox(frame_PIL=frame_PIL, bbox=obj_bbox))
                obj_NL_relation = api_response_base64picture(client= qwen_client, model = Model_dic["QWEN-VL-MAX"], 
                                                              prompt= prompt, base64_picture = frame_with_bbox)
                
                # print("Step 2.iii response [obj_NL_relation]:", obj_NL_relation)

                prompt = f"""From the given sentence, the task is to extract meaningful triplets formed as <subject, predicate, object>. Note that the subject is the entity or noun that performs the action or is being described, and the object is the entity or noun that is affected by the action or is receiving the action. The predicate is a verb or adjective without auxiliary verb, and is represented without the tense (e.g., are, being).
If "object" in triplet has a plural form, then your answer should be its singular form.
Let's take a few examples to understand how to extract meaningful triplets. 
Question: Given the sentence "a slice of bread is covered with a sour cream and quacamole," extract meaningful triplets. The \"subject\" of the first triplet must be bread. Answer: Meaningful triplets are <subject:d, predicate:covered with, object:sour cream>, <subject:bread, predicate:covered with, object:guacamole>. 
Question: Given the sentence "A beautiful woman walking a dog on top of a beach," extract meaningful triplets. The \"subject\" of the first triplet must be woman. Answer: Meaningful triplets are <subject:woman, predicate:walking with, object:dog>, <subject:woman, predicate:on, object:beach>, <subject:dog, predicate:on, object:beach>. 
Question: Given the sentence "Four clock sitting on a floor next to a woman's feet," extract meaningful triplets. The \"subject\" of the first triplet must be clock. Answer: Meaningful triplets are <subject:clock, predicate:sitting on, object:floor>, <subject:clock, predicate:next to, object:feet>. 
Question: Given the sentence "One person sits in a chair looking at her phone while another rests on the couch," extract meaningful triplets. The \"subject\" of the first triplet must be person. Answer: Meaningful triplets are <subject:person, predicate:sits in, object:chair>, <subject:person, predicate:looking at, object:phone>, <subject:person, predicate:rests on, object:couch>. 
Question: Given the sentence "A lady and a child near a park bench with kites and ducks flying in the sky and on the ground," extract meaningful triplets. The \"subject\" of the first triplet must be lady. Answer: Meaningful triplets are <subject:lady, predicate:near, object:park bench>, <subject:child, predicate:near, object:park bench>, <subject:kites, predicate:flying in, object:sky>, <subject:ducks, predicate:on, object:ground>. 
Question: Given the sentence "Two men sit on a bench near the sidewalk and one of them talks on a cell phone," extract meaningful triplets. The \"subject\" of the first triplet must be men. Answer: Meaningful triplets are <subject:men, predicate:sit on, object:bench>, <subject:bench, predicate:near, object:sidewalk>, <subject:man, predicate:talks on, object:phone>. 
Please answer the following question. 
"""
                full_prompt = prompt + f"Given the sentence \"{obj_NL_relation}\", extract meaningful triplets. The \"subject\" of the first triplet must be {obj_label}. Your answer must strictly follow the pattern in examples. Answer: "
                response = api_response_textonly(client=qwen_client, model=Model_dic["QWEN-PLUS"], prompt=full_prompt)               
                # print("Step 2.iii response [obj_NL_triplet]:", response)
                triplets = stsg.extract_triplets(response)
                # print(triplets)

                # 主要关系
                for row in triplets:
                    if row["subject"] == obj_label and row["object"] in obj_labels:
                        if obj_labels.count(row["object"]) > 1:

                            relevant_obj_list = {"bboxes":[], "labels":[], "reference":[]}
                            prob_obj_list = []

                            for idx in range(len(frame_od_list["labels"])):
                                if frame_od_list["labels"][idx] == row["subject"] and frame_od_list["reference"][idx] == obj_distinct_label:
                                    relevant_obj_list["bboxes"].append(frame_od_list["bboxes"][idx])
                                    relevant_obj_list["labels"].append(frame_od_list["labels"][idx])
                                    relevant_obj_list["reference"].append(frame_od_list["reference"][idx])
                                elif frame_od_list["labels"][idx] == row["object"]:
                                    relevant_obj_list["bboxes"].append(frame_od_list["bboxes"][idx])
                                    relevant_obj_list["labels"].append(frame_od_list["labels"][idx])
                                    relevant_obj_list["reference"].append(frame_od_list["reference"][idx])
                                    prob_obj_list.append(row["object"] + '#' + str(frame_od_list["reference"][idx]))

                            prompt = f"""Watch the picture and complete the task: 
Given a triplet <subject: {obj_label +'#'+str(obj_distinct_label)}, predicate: {row["predicate"]}, object{row["object"]}> representing the relationship between {obj_label + '#' + str(obj_distinct_label)} and other objects,
find the most suitable {row["object"]} for the object of triplet. Your answer should be the distinct label showed on image among {prob_obj_list}.
"""  

                            frame_with_labels = frame_add_bboxes_and_labels_b64(
                                frame_base64,
                                relevant_obj_list,
                                save_frames = True,
                                save_name = os.path.join(example_frames_dir, f"bbox_frame_{frame_no}.jpg")
                            )
                            frame_with_labels_b64 = cv2_to_base64(frame_with_labels)
                            # print("Multiple Candidate Prompt:", prompt)
                            response = api_response_base64picture(client = qwen_client, model = Model_dic["QWEN-VL-MAX"], prompt = prompt, base64_picture = frame_with_labels_b64)
                            # print("Triplet Object Response:", response)
                            for obj_1 in frame_graph.objects:
                                if (obj_1.object_name + '#' + str(obj_1.distinct_label)) in response:
                                    Rel = stsg.Relationship(
                                        segment_id = segment_id, 
                                        frame_id = frame_id, 
                                        relation_id = len(frame_graph.relations),
                                        subject = obj.object_id,
                                        predicate = row["predicate"],
                                        object = obj_1.object_id,
                                        time = frame_graph.time
                                    )
                                    frame_graph.add_relation(Rel)

                        else:
                            Rel = stsg.Relationship(
                                segment_id = segment_id, 
                                frame_id = frame_id, 
                                relation_id = len(frame_graph.relations),
                                subject = obj.object_id,
                                predicate = row["predicate"],
                                object = stsg.get_object_by_label(row["object"], frame_graph.objects).object_id,
                                time = frame_graph.time
                            )
                            frame_graph.add_relation(Rel)
                    elif row["subject"] == obj_label and row["object"] not in obj_labels:
                        if row["object"] != "image":
                            obj.add_attribute(stsg.Attribute(row["predicate"], row["object"]))
                    elif obj.get_correspond_attributes_id(row["subject"]) != -1:
                        if row["object"] != "image":
                            obj.add_attribute(stsg.Attribute(row["predicate"], row["object"], obj.get_correspond_attributes_id(row["subject"])))
                            
            # print("frame_graph:", frame_graph)


### For Segment-level STSG Generation

            if frame_id == 0:
                segment_graph.objects = frame_graph.objects
                segment_graph.relations = frame_graph.relations
            else:
                replace_list = np.full(len(frame_graph.objects), -1)
                seg_obj_list = segment_graph.objects
                for new_obj in frame_graph.objects:
                    for alt_obj in reversed(seg_obj_list): #注: 防止存在tracklet的情况下联系到老的元素
                        # 判断物品的继承
                        # 是同一元素
                        if new_obj.distinct_label == alt_obj.distinct_label: # if new_obj.object_name == alt_obj.object_name: 
                            
                            object_reference_label = all_object_list[new_obj.distinct_label]["label"]
                            new_bbox = new_obj.bbox[-1]
                            alt_bbox = alt_obj.bbox[-1]

                            obj_start_time = alt_obj.time[-1]
                            obj_end_time = new_obj.time[-1]
                            obj_start_frame = int(alt_obj.time[-1] * fps)
                            obj_end_frame = int(new_obj.time[-1] * fps)

                            sampling_number = 8
                            sampling_list = [int(obj_start_frame + (obj_end_frame-obj_start_frame)/(sampling_number-1)*i) for i in range(sampling_number)]
                            sampling_frame_list_b64 = videoframes_to_base64(video_path, sampling_list)
                            
                            sampling_start_frame = sampling_frame_list_b64[0]
                            sampling_end_frame = sampling_frame_list_b64[-1]

                            sampling_start_frame_cv2 = frame_add_bbox_b64(sampling_start_frame, alt_bbox)
                            sampling_end_frame_cv2 = frame_add_bbox_b64(sampling_end_frame, new_bbox)

                            sampling_frame_list_b64[0] = cv2_to_base64(sampling_start_frame_cv2)
                            sampling_frame_list_b64[-1] = cv2_to_base64(sampling_end_frame_cv2)
                            

                            prompt = f"""For the given video, {object_reference_label} highlighted in the start of video and {object_reference_label} highlighted in the end of video is the same object. Determine wheter the spatial position of this object has changed in video. Just output 'Yes' or 'No'.""" # prompt需要优化
                            response = api_response_base64piclist(client=qwen_client, model=Model_dic["QWEN-VL-MAX"], prompt=prompt, base64_picture_list=sampling_frame_list_b64)

                            # print("Position Change:", response)
                                # 位置发生了变化, 元素分割构造动边
                            if "yes" in response.lower():
                                tracklet = stsg.find_tracklet(alt_obj.object_id, segment_graph.tracklets)
                                if tracklet == None:
                                    tracklet = stsg.Tracklet(
                                        segment_id=segment_id,
                                        tracklet_id=len(segment_graph.tracklets),
                                        obj1=alt_obj.object_id, 
                                        obj2=len(segment_graph.objects)
                                    )
                                    segment_graph.add_tracklet(tracklet)
                                else:
                                    tracklet.add_object(len(segment_graph.objects))
                                    
                                # Step 3.ii.a 将元素的动作整合
                                prompt = f"""Given the same object {object_reference_label} highlighted in the start of video and {object_reference_label} highlighted in the end of video , describe the action of this object in the video. """
                                NL_action = api_response_base64piclist(client=qwen_client, model=Model_dic["QWEN-VL-MAX"], prompt=prompt, base64_picture_list=sampling_frame_list_b64)
                                # print("Step 3.ii.a:", NL_action)

                                # Step 3.ii.b 将动作抽象为字典格式
                                prompt = f"""For the given sentence, the task is to extract meaningful action for the given object. The action should be described using a verb or verb phrase.
Let's take a few examples to understand how to extract.
Question: Given the sentence "The man is walking on the street." Extract meaningful action for the object "man".
Answer: {{"action":["walking on the street"]}}
Question: Given the sentence "The cat is jumping on the bed." Extract meaningful action for the object "cat".
Answer: {{"action":["jumping on the bed"]}}
Question: Given the sentence "The woman is sitting on the chair and holding a bottle." Extract meaningful action for the object "woman".
Answer: {{"action":["sitting on the chair", "holding a bottle"]}}
"""
                                full_prompt = prompt + f"Given the sentence \"{NL_action}\", extract meaningful action for the object \"{object_reference_label}\"."
                                dict_action_raw = api_response_textonly(client=qwen_client, model=Model_dic["QWEN-PLUS"], prompt=full_prompt)
                                dict_action_raw = parsing_str(dict_action_raw)
                                # print("Step 3.ii.b:", dict_action_raw)

                                # Step 3.ii.c 将动作与目标分离
                                if '{' in dict_action_raw and '}' in dict_action_raw:
                                    dict_action = '{' + dict_action_raw.split('{')[1].split('}')[0] + '}'
                                    try:
                                        filtered_action = json.loads(dict_action)["action"]
                                    except json.JSONDecodeError:
                                        filtered_action = re.findall(r'\"action\":[\"(.*?)\"]', dict_action_raw)
                                else:
                                    filtered_action = re.findall(r'\"action\":[\"(.*?)\"]', dict_action_raw)
                                    
                                for action in filtered_action:
                                    prompt = f"""For the given action "{action}", extract the predicate and object. The predicate is a verb or verb phrase, and the object is the entity or noun that is affected by the action
Let's take a few examples to understand how to extract.
Question: Given the action "walking", extract the predicate and object.
Answer: ["predicate":"walking", "object":"None"]
Question: Given the action "jumping on the bed", extract the predicate and object.
Answer: ["predicate":"jumping on", "object":"bed"]
Question: Given the action "sitting on the chair", extract the predicate and object.
Answer: ["predicate":"sitting on", "object":"chair"]
"""
                                    full_prompt = prompt + f"Given the action \"{action}\", extract the predicate and object. Answer: "
                                    response = api_response_textonly(client=qwen_client, model=Model_dic["QWEN-PLUS"], prompt=full_prompt)
                                    response = parsing_str(response)

                                    # print("Step 3.ii.c:", response)
                                    predicate_match = re.search(r'\"predicate\":\"(.*?)\"', response)
                                    object_match = re.search(r'\"object\":\"(.*?)\"', response)
                                    
                                    if predicate_match and object_match:
                                        # print("New Action!!!")
                                        predicate_ = predicate_match.group(1)
                                        object_ = object_match.group(1)
                                        Act = stsg.Action(
                                            segment_id=segment_id,
                                            action_id=len(segment_graph.actions),
                                            action_description=NL_action,
                                            start_time=obj_start_time,
                                            end_time=obj_end_time,
                                            tracklet_id=tracklet.tracklet_id,
                                            subject=alt_obj.object_id,
                                            predicate=predicate_,
                                            object =object_
                                        )
                                        alt_obj.add_action(Act)
                                        segment_graph.add_action(Act)
                                        # 直接复制自getSTSG_mistral，需要更改
                                        # for obj_ in segment_graph.objects:
                                        #     if obj_.distinct_label == object_:
                                        #         obj_.add_action(Act)
                                break
                            else:
                                alt_obj.time.extend(new_obj.time)
                                alt_obj.bbox.extend(new_obj.bbox)
                                replace_list[new_obj.object_id] = alt_obj.object_id
                                break
                        # 判断物品的演化关系
                    # print(new_obj)

                    if replace_list[new_obj.object_id] == -1:

                        # print("Add new object:", all_object_list[new_obj.distinct_label]["label"])

                        replace_list[new_obj.object_id] = len(segment_graph.objects)
                        new_obj.object_id = len(segment_graph.objects)
                        segment_graph.objects.append(new_obj)

                for rel in frame_graph.relations:
                    rel.subject = replace_list[rel.subject]
                    rel.object = replace_list[rel.object]
                    if stsg.rel_exists(rel, segment_graph.relations):
                        rel0 = stsg.get_rel_by_subject_object_predicate(rel.subject,rel.object,rel.predicate,segment_graph.relations)
                        rel0.time.append(rel.time[-1])
                    else:
                        segment_graph.relations.append(rel)
                        
            print("===> segment_graph")
        
        sgs.append(segment_graph)
    
    if len(sgs)==0:
        continue
    video_graph = stsg.VideoSTSG(video_path, sgs[0].start_time, sgs[-1].end_time)

    for i in range(0, len(sgs)-1):
        video_graph.add_segment(sgs[i])
        video_graph.add_event(sgs[i].event)
        for seg_action in sgs[i].actions:
            video_graph.add_action(seg_action)

        sg1 = sgs[i]
        sg2 = sgs[i+1]

        for obj_1 in sg1.objects:
            for obj_2 in sg2.objects:
                if obj_1.distinct_label == obj_2.distinct_label:
                    refer = stsg.Reference(
                        reference_id=len(video_graph.reference),
                        obj1=obj_1,
                        obj2=obj_2
                    )
                    video_graph.add_reference(refer)

    if len(sgs) > 0:
        video_graph.add_segment(sgs[-1])
        video_graph.add_event(sgs[-1].event)
        for seg_action in sgs[-1].actions:
            video_graph.add_action(seg_action)

        count = count+1

    output_video_list = []
    for obj in all_object_list:
        prompt = f"""Summarize the characteristics of {obj["label"]} in a concise sentence starting with "{obj["label"]}" and without referencing other items, your answer should based on {obj["label"]} and its various attributes throughout the video: {obj["attributes"]}."""
        response = api_response_textonly(client=qwen_client, model=Model_dic["QWEN-PLUS"], prompt=prompt)
        print("##prompt:", prompt)
        print("##response:", response)

        new_dict = {
            "index": obj["index"],
            "label": obj["label"],
            "start_frame": obj["start_frame"],
            "attributes": obj["attributes"],
            "NL_summary": response
        }

        output_video_list.append(new_dict)
    # print("===> Video_graph:", video_graph)
    output_stsg_dict = {
        "video": video_path,
        "fps": fps,
        "dimensions": dim, 
        "object_list": output_video_list,
        "video_graph": video_graph.to_dict()
    }

    with open(write_path, 'a') as f:
        json.dump(output_stsg_dict, f)

    # raise NotImplementedError("输出和保存")
