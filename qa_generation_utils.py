import math
import random
from utils.api_utils import *
import json
import torchvision.ops as tvops
import torch
from collections import deque
import copy
import random
from itertools import combinations

def seconds_to_mmss(seconds):
    # 计算分钟数
    minutes = int(seconds // 60)
    # 计算剩余秒数
    remaining_seconds = seconds % 60
    # 正确格式化为 MM:SS.ff
    return f"{minutes:02d}:{remaining_seconds:05.2f}"

def object_action_to_description(target_object):
    action_description_list = []
    for action in target_object["actions"]:
        new_dict = {
            "start_time": seconds_to_mmss(action["start_time"]),
            "end_time": seconds_to_mmss(action["end_time"]),
            "action_description": action["action_description"]
        }
        action_description_list.append(new_dict)
    
    return action_description_list



class QRA_pair:
    def __init__(self, question, reasoning, answer, source, start_frame = None, end_frame = None):
        self.question = question
        self.reasoning = reasoning
        self.answer = answer
        self.source = source
        self.start_frame = start_frame
        self.end_frame = end_frame
    def __str__(self):
        return f'{{"question": "{self.question}", "reasoning": "{self.reasoning}", "answer": "{self.answer}", "source": {self.source}, "start_frame": {self.start_frame}, "end_frame": {self.end_frame}}}'


# QRCA_pair => question, reasoning, choices, answer
class QRCA_pair:
    def __init__(self, question, reasoning, choices, answer):
        self.question = question
        self.reasoning = reasoning
        self.choices = choices
        self.answer = answer

# 根据视频中元素的唯一编号与目标帧的编号生成目标元素在画面中的相对位置

def search_direction(video_data, obj_index, target_frame):
    '''
    obj_index: The distinct label of target obj
    '''
    search_flag = 0
    bbox = None
    for segment in video_data["video_graph"]["segments"]:
        if target_frame >= round(segment["start_time"] * video_data["fps"]) and target_frame <=  round(segment["end_time"] * video_data["fps"]): # In this segment.
            for obj in segment["objects"]:
                if obj["distinct_label"] == obj_index: # Find the target
                    for i in range(len(obj["time"])):
                        if round(obj["time"][i] * video_data["fps"]) == target_frame:
                            search_flag = 1
                            bbox = obj["bbox"][i]
                            break
                    if search_flag:
                        break
            if search_flag:
                break
    w = video_data["dimensions"][0]
    h = video_data["dimensions"][1]
    if bbox is not None:
        middle_point_x = (bbox[0] + bbox[2]) / 2
        middle_point_y = (bbox[1] + bbox[3]) / 2
        if middle_point_x > 0.4 * w  and middle_point_x < 0.6 * w and middle_point_y > 0.4 * h and middle_point_y < 0.6 * h:
            return "middle of"
        elif middle_point_x < 0.5 * w and middle_point_y < 0.5 * h:
            return "upper left of"
        elif middle_point_x >= 0.5 * w and middle_point_y < 0.5 * h:
            return "upper right of"
        elif middle_point_x < 0.5 * w and middle_point_y >= 0.5 * h:
            return "lower left of"
        elif middle_point_x >= 0.5 * w and middle_point_y >= 0.5 * h:
            return "lower right of"
        else:
            return None
    return None

def search_bbox(video_data, obj_index, target_frame):
    search_flag = 0
    bbox = None
    for segment in video_data["video_graph"]["segments"]:
        if target_frame >= round(segment["start_time"] * video_data["fps"]) and target_frame <=  round(segment["end_time"] * video_data["fps"]): # In this segment.
            for obj in segment["objects"]:
                if obj["distinct_label"] == obj_index: # Find the target
                    for i in range(len(obj["time"])):
                        if round(obj["time"][i] * video_data["fps"]) == target_frame:
                            search_flag = 1
                            bbox = obj["bbox"][i]
                            break
                    if search_flag:
                        break
            if search_flag:
                break
    if bbox is not None:
        return bbox
    return None

def get_obj_feature(video_data, obj_index, target_frame):
    search_flag = 0
    bbox = None
    object_data = None
    for segment in video_data["video_graph"]["segments"]:
        if target_frame >= round(segment["start_time"] * video_data["fps"]) and target_frame <=  round(segment["end_time"] * video_data["fps"]): # In this segment.
            for obj in segment["objects"]:
                if obj["distinct_label"] == obj_index: # Find the target
                    for i in range(len(obj["time"])):
                        if round(obj["time"][i] * video_data["fps"]) == target_frame:
                            search_flag = 1
                            bbox = obj["bbox"][i]
                            object_data = obj
                            break
                    if search_flag:
                        break
            if search_flag:
                break 
    object_description = ''
    attr_description = ''
    if bbox is not None and object_data is not None:
        for attr in object_data['attributes']:
            if attr['attribute_value'] is not None:
                attr_description += f" [{attr['attribute_name']}: {attr['attribute_value']}]"
            else:
                attr_description += f" [{attr['attribute_name']}]"
        if attr_description:
            object_description += f"{attr_description}"
        return object_description
    else:
        return None 


def generate_qa_via_video_object_iteration(video_data, client, model):
    object_dict = {}
    qra_list = []
    video_stsg = video_data["video_graph"]
    fps = video_data["fps"]

    # video_data["object_list"] = [{"index": 0, "label": xxx, "start_frame": xxx}, {...}, ...]

    for obj in video_data["object_list"]:
        if obj["label"] in object_dict:
            object_dict[obj["label"]].append(obj["index"])
        else:
            object_dict[obj["label"]] = [obj["index"]]
    
    # generate Video-Stage Temporal Counting
    for key in object_dict:
        # generate Video-Stage Temporal Counting
        if len(object_dict[key]) > 1:
            annotation = {}
            count = 0

            # Iteration on Video Object
            for distinct_index in object_dict[key]:
                count += 1
                annotation[f"{key}#{count}"] = []
                for segment_stsg in video_stsg["segments"]:
                    for segment_obj in segment_stsg["objects"]:
                        if segment_obj["distinct_label"] == distinct_index:
                            time_and_position = {}
                            for time_ in segment_obj["time"]:
                                time_and_position[seconds_to_mmss(time_)] = f"{search_direction(video_data= video_data, obj_index=distinct_index, target_frame = round(time_ * fps))} video"
                            filtered_attributes = []
                            for obj_attribute in segment_obj["attributes"]:
                                if obj_attribute["attachmant"] == -1:
                                    attr_name = obj_attribute["attribute_name"]
                                    attr_value = obj_attribute["attribute_value"]
                                    filtered_attributes.append(f"{{'name':{attr_name}, 'value':{attr_value} }}")
                            new_dict = {
                                "time_and_position": time_and_position,
                                "attributes": filtered_attributes
                            }
                            annotation[f"{key}#{count}"].append(new_dict)
            # print(annotation)
            
            # 消除重复元素
            prompt = f"""You are a data analysis assistant specialized in analyzing and categorizing object records from videos. \
Given time intervals and feature records about multiple objects, determine how many distinct objects are present in the video and group the records that belong to the same object, considering that records from different categories may refer to the same object. \
The output should be in a JSON file as a Python dictionary, where the keys represent each unique object (e.g., "distinct_{key}#1") and the values are all the record categories representing that object (e.g., ["{key}#1", "{key}#2"]).'

### Output Format:
1. Your output should be formed in a JSON file.
2. Only provide the Python dictionary string.

EXAMPLE:
{{
    "distinct_{key}#1": ["{key}#1", "{key}#2"],
    "distinct_{key}#2": ["{key}#3"],
    "distinct_{key}#3": ["{key}#4”]
}}

### Object Record:
{annotation}
"""   
            response = api_response_textonly(client=client, model=model, prompt=prompt)
            response = response.replace("```json", "")
            response = response.replace("```", "")
            try:
                response_json = json.loads(response)
            except json.decoder.JSONDecodeError as e:
                print("Decode Error For:", response)
                continue

            processed_annotation = {}

            if len(response_json) >= 1:

                if len(response_json) == 1 and random.random() > 0.25:
                    continue
                print(key, ":", len(response_json))
                raw_answer = len(response_json)
                raw_question = f"How many distinct {key} appears in the video?"
                for distinct_key in response_json:
                    for raw_key in response_json[distinct_key]:
                        if raw_key not in annotation:
                            continue
                        if distinct_key in processed_annotation:
                            processed_annotation[distinct_key].extend(annotation[raw_key])
                        else:
                            processed_annotation[distinct_key] = annotation[raw_key]
                
                
                # Video Temporal Counting (Object)
                prompt = f"""You are an educational assistant specialized in designing problems and providing solutions and reasoning. \
Given time intervals and feature records about multiple objects, a short-answer question and its correct answer, \
your task is to transform it into a multiple-choice question format and generate the problem-solving process step by step.

### Attention:
1. Your reasoning step should be no more than {min((raw_answer + 2), 6)} steps.
2. Your reasoning process must not directly use labels in the object record(eg. distinct_{key}#1), instead you should guide students to find relevant information for the answer from the original video.
3. Your reasoning process must contain the overall feature of each distinct object and its first time appears in the video.
4. The last step of your reasoning process must end with "so the correct choice is ...".
5. The correct Choice should be {random.choice(['A', 'B', 'C', 'D'])}

The output should be in a JSON file as a Python dictionary followed output format below:

### Output Format:
{{
    "question": "How many distinct {key} appears in the video?",
    "answer":"[The correct choice]",
    "choices": {{"A": "...", "B": "...", "C": "...", D: "..."}},
    "reasoning": {{
            "step1": "[Your Reasoning]",
            "step2": "[Your Reasoning]", 
            ...
            }}
}}

### Object Record:
{processed_annotation}

### Question:
{raw_question}

### Answer:
{raw_answer}
"""
                response = api_response_textonly(client=client, model=model, prompt=prompt)
                # print(response)

                response = response.replace("```json", "")
                response = response.replace("```", "")
                try:
                    question_json = json.loads(response)
                    question_json["source"] = video_data["video"]
                    question_json["type"] = "VTC-O"
                    question_json["generation_message"] = {"obj": key}
                    qra_list.append(question_json)
                except json.decoder.JSONDecodeError as e:
                    print("Decode Error For:", response)

                # Temporal Object Perception
                prompt = f"""You are an educational assistant specialized in designing problems and providing solutions and reasoning. \
Given time intervals and feature records about multiple objects, your task is to design a multiple-choice question \
about the order of appearance of these elements in the video and generate the problem solving process step by step.

### Attention
    1. The question must follow the shape like "Which of the following {key} appears first in the video?" or "Which of the following {key} disappears last in the video?"
    2. There may be multiple objects of the same type, so the choice must reflect the unique feature of each object.
    3. Different choices of the question should be distinguishable. Including correct answer, wrong answer with wrong order, and wrong answer with object didn't appear in video.
    4. Your reasoning process must not directly use labels in the object record(eg. distinct_{key}#1) or mention the existence of object record, instead you should guide students to find relevant information for the answer from the original video.
    5. Your reasoning process should utilize time data in object record.
    6. The last step of your reasoning process must end with "so the correct choice is ...".

The output should be in a JSON file as a Python dictionary followed output format below:

### Output Format:
{{
    "question": "Which of the following {key} ...",
    "choices": {{"A": "...", "B": "...", "C": "...", "D": ...}},
    "answer": "[the correct choices]",
    "reasoning": {{
        "step1": "[your reasoning]"
        ...
    }}
}}

### Object Record:
{processed_annotation}
"""
                response = api_response_textonly(client=client, model=model, prompt=prompt)
                # print(response)
                response = response.replace("```json", "")
                response = response.replace("```", "")
                try:
                    question_json = json.loads(response)
                    question_json["source"] = video_data["video"]
                    question_json["type"] = "TOP"
                    question_json["generation_message"] = {"obj": key}
                    qra_list.append(question_json)
                except json.decoder.JSONDecodeError as e:
                    print("Decode Error For:", response)
    return qra_list

def generate_qa_via_video_action_iteration(video_data, client, model):
    object_dict = {}
    qra_list = []
    video_stsg = video_data["video_graph"]
    for action in video_stsg["actions"]:
        # get_distinct_id: video_stsg.segments[segment_id].objects[object_id].distinct_label
        distinct_id = video_stsg["segments"][action["segment_id"]]["objects"][action["subject"]]["distinct_label"]
        new_dict = {
            "start_time": seconds_to_mmss(action["start_time"]),
            "end_time": seconds_to_mmss(action["end_time"]),
            "action_description": action["action_description"]
        }
        if distinct_id in object_dict:
            object_dict[distinct_id].append(new_dict)
        else:
            object_dict[distinct_id] = [new_dict]
    
    for key in object_dict:
        if len(object_dict[key]) <= 2:
            continue
        distinct_label = video_data["object_list"][key]["label"]
        NL_summary = video_data["object_list"][key]["NL_summary"]

        # Temporal Action Perception
        prompt = f"""You are an educational assistant specialized in designing problems and providing solutions and reasoning. \
Given time and action records about a {distinct_label}, your task is to design a multiple-choice question about the temporal relation of these actions, \
and generate the problem-solving process step by step.

### Attention
    1. There may be multiple objects of the same type, so the question stem must reflect the unique feature of the target object.
    2. The description of the target object's features should be concise and focus on static characteristics.
    3. Different choices of the question should be distinguishable. Including correct answer, wrong answer with action in records, wrong answer with action didn't happen.
    4. Your reasoning process must not directly use labels in the action record or mention the existence of the action record, instead you should guide students to find relevant information for the answer from the original video.
    5. Your reasoning process should contain time data of relevant actions.
    6. The last step of your reasoning process must end with "so the correct choice is ...".

The output should be in a JSON file as a Python dictionary followed output format below:

### Output Format:
{{
    "question": "For [describe target object], which action happens first / last ?",
    "choices": {{"A": "...", "B": "...", "C": "...", "D": ...}},
    "answer": "[the correct choices]",
    "reasoning": {{
        "step1": "[your reasoning]"
        ...
    }}
}}

### Objcet Feature Describe:
{NL_summary}

### Action Records:
{object_dict[key]}
"""
        response = api_response_textonly(client=client, model=model, prompt=prompt)
        # print(response)
        # post processing
        response = response.replace("```json", "")
        response = response.replace("```", "")
        try:      
            question_json = json.loads(response)
            question_json["source"] = video_data["video"]
            question_json["type"] = "TAP" 
            question_json["generation_message"] = {"obj": distinct_label}
            qra_list.append(question_json)
        except json.decoder.JSONDecodeError as e:
                print("Decode Error For:", response)

        # Temporal Anomaly Detection
        prompt = f"""You are an educational assistant specialized in designing problems and providing solutions and reasoning. \
Given time and action records about a {distinct_label}, your task is to design a multiple-choice question to challenge the test-taker's memory ability.

### Attention:
    1. There may be multiple objects of the same type, so the question stem must reflect the unique feature of the target object.
    2. Don't use features like "highlighted in video" or "in a green box" in description of the target object.
    3. The question stem must follow the shape like "Which event didn't happen to [target object with description]?"
    4. Different choices of the question should be distinguishable. Including an choice with a fictional event that did not occur in the video, and three other choices with actions in action records.

The output should be in a JSON file as a Python dictionary followed output format below:

### Output Format:
{{
    "question": "Which event [happened / didn't happen] to [target object with description]",
    "choices": {{"A": "...", "B": "...", "C": "...", "D": ...}},
    "answer": "[the correct choices]"
}}
### Objcet Feature Describe:
{NL_summary}

### Action Records:
{object_dict[key]}
"""
        response = api_response_textonly(client=client, model=model, prompt=prompt)
        # print(response)
        # post processing
        response = response.replace("```json", "")
        response = response.replace("```", "")
        try:    
            question_json = json.loads(response)
            question_json["reasoning"] = None
            question_json["source"] = video_data["video"]
            question_json["type"] = "TAD" 
            question_json["generation_message"] = {"obj": distinct_label}
            qra_list.append(question_json)
        except json.decoder.JSONDecodeError as e:
            print("Decode Error For:", response)
    
    all_action_dict = {}
    all_object_categories = []
    for key in object_dict:
        distinct_label = video_data["object_list"][key]["label"]
        NL_summary = video_data["object_list"][key]["NL_summary"]
        new_dict = {
            "label": distinct_label,
            "NL_summary": NL_summary,
            "Events": object_dict[key]
        }
        if f"object#{key}" in all_action_dict:
            all_action_dict[f"object#{key}"].append(new_dict)
        else:
            all_action_dict[f"object#{key}"] = [new_dict]

        if distinct_label not in all_object_categories:
            all_object_categories.append(distinct_label)

    all_action_description = ''

    for key in all_action_dict: # iteration on each distinct objects
        all_action_description += f"{key} is a {all_action_dict[key][0]['label']}."
        all_action_description += f" {all_action_dict[key][0]['NL_summary']}"
        obj_action_dict = []
        for obj_description in all_action_dict[key]:
            for obj_action in obj_description['Events']:
                new_action_str = f"Between {obj_action['start_time']} and {obj_action['end_time']}, {obj_action['action_description']}"
                if new_action_str not in obj_action_dict:
                    obj_action_dict.append(new_action_str)
                    all_action_description += f" {new_action_str}"
        all_action_description += '\n'

    for label in all_object_categories:
        # Temporal Event Grounding
        prompt = f"""You are an educational assistant specialized in designing problems and providing solutions and reasoning. \
Given object and event records of the video, your task is to design a multiple-choice question to challenge the test-taker's memory ability.

### Attention:
    1. Don't use features like "highlighted in video" or "in a green box" in description of event.
    2. The correct choice must be an event happend in a {label}, while wrong choices can be an event happened in anything.
    3. The question stem must follow the shape like "Which event occurs between [time1] and [time2]?", correct choice must occur between [time1] and [time2].
    4. Different choices of the question should be distinguishable. Several alternative wrong choices include fictional event that did not occur in the video and event that did not happend between [time1] and [time2].

The output should be in a JSON file as a Python dictionary followed output format below:

### Output Format:
{{
    "question": "Which event occurs between [time1] and [time2]?",
    "choices": {{"A": "...", "B": "...", "C": "...", "D": ...}},
    "answer": "[the correct choices]",
}}
### Objcet Feature Describe:
{all_action_description}
"""
        response = api_response_textonly(client=client, model=model, prompt=prompt)
        response = response.replace("```json", "")
        response = response.replace("```", "")
        try:     
            question_json = json.loads(response)
            question_json["reasoning"] = None
            question_json["source"] = video_data["video"]
            question_json["type"] = "TEG" 
            question_json["generation_message"] = {"obj": label}
            qra_list.append(question_json)
        except json.decoder.JSONDecodeError as e:
            print("Decode Error For:", response)
    
    for label in all_object_categories:
        # Temporal Event Perception
        prompt = f"""You are an educational assistant specialized in designing problems and providing solutions and reasoning. \
Given object and event records of the video, your task is to design a multiple-choice question to challenge the test-taker's memory ability.

### Attention:
    1. Don't use features like "highlighted in video" or "in a green box" in description of event.
    2. The question stem must follow the shape like "Which event didn't occur in video?"
    3. Different choices of the question should be distinguishable. The correct choice should be a fictional event that did not occur on {label}. Other choice must be events in object and event records.

The output should be in a JSON file as a Python dictionary followed output format below:

### Output Format:
{{
    "question": "Which event didn't occur in video?",
    "choices": {{"A": "...", "B": "...", "C": "...", "D": ...}},
    "answer": "[the correct choices]",
}}
### Objcet Feature Describe:
{all_action_description}
"""
        response = api_response_textonly(client=client, model=model, prompt=prompt)
        response = response.replace("```json", "")
        response = response.replace("```", "")
        try:   
            question_json = json.loads(response)
            question_json["reasoning"] = None
            question_json["source"] = video_data["video"]
            question_json["type"] = "TEP" 
            question_json["generation_message"] = {"obj": label}
            qra_list.append(question_json)
        except json.decoder.JSONDecodeError as e:
            print("Decode Error For:", response)
    return qra_list

def generate_qa_via_video_segment_iteration(video_data, client, model):
    object_dict = {}
    qra_list = []
    video_stsg = video_data["video_graph"]
    fps = video_data["fps"]
    available_choices = ['A', 'B', 'C', 'D']

    for segment in video_stsg["segments"]:

        object_dict = {}
        distinct_id_dict = {}
        object_action_dict = {}

        segment_obj_citation = {}
        segment_obj_annotation = {}
        segment_obj_action = {}

        frame_obj_dict = {}

        seg_start_time = seconds_to_mmss(segment["start_time"])
        seg_end_time = seconds_to_mmss(segment["end_time"])
        seg_total_time = segment["end_time"] - segment["start_time"]

        for segment_obj in segment["objects"]:

            obj_name = segment_obj["object_name"]
            obj_id = segment_obj["object_id"]
            obj_did = segment_obj["distinct_label"]
            obj_action = object_action_to_description(segment_obj)

            if obj_name in object_dict:
                if obj_did not in object_dict[obj_name]:
                    object_dict[obj_name].append(obj_did)
            else:
                object_dict[obj_name] = [ obj_did ]
            
            if obj_did in distinct_id_dict:
                distinct_id_dict[obj_did].append(obj_id)
            else:
                distinct_id_dict[obj_did] = [obj_id]

            if obj_did in object_action_dict:
                object_action_dict[obj_did].extend(obj_action)
            else:
                object_action_dict[obj_did] = obj_action
            
            # 为每个关键帧注册物品列表
            for i in range(0, len(segment_obj["time"])):
                obj_time = segment_obj["time"][i]
                temp_frame = round(obj_time * fps)
                if temp_frame in frame_obj_dict:
                    if obj_name not in frame_obj_dict[temp_frame]: #该类物品未注册
                        frame_obj_dict[temp_frame][obj_name] = {}
                        frame_obj_dict[temp_frame][obj_name][obj_did] = segment_obj
                        frame_obj_dict[temp_frame][obj_name][obj_did]["current_index"] = i
                    elif obj_did not in frame_obj_dict[temp_frame][obj_name]: #该物品未注册
                        frame_obj_dict[temp_frame][obj_name][obj_did] = segment_obj
                        frame_obj_dict[temp_frame][obj_name][obj_did]["current_index"] = i
                else:
                    frame_obj_dict[temp_frame] = {}
                    frame_obj_dict[temp_frame][obj_name] = {}
                    frame_obj_dict[temp_frame][obj_name][obj_did] = segment_obj
                    frame_obj_dict[temp_frame][obj_name][obj_did]["current_index"] = i


        # generate frame-class question(test)
        for frame in frame_obj_dict:
            for obj_class in frame_obj_dict[frame]:
                frame_time = seconds_to_mmss(frame / fps)
                
                if len(frame_obj_dict[frame][obj_class]) <= 1 and random.random() > 0.1:
                    continue
                elif len(frame_obj_dict[frame][obj_class]) > 1 and random.random() > 0.5:
                    continue
                
                prompt = f"""You are an educational assistant specialized in designing problems, given question stem and correct answer, your task is to rewrite it to a multiple-choice question.
The output should be in a JSON file as a Python dictionary followed output format below:

### Output Format:
{{
    "question": "How many distinct {obj_class} appear at {frame_time}?",
    "choices": {{"A": "...", "B": "...", "C": "...", "D": ...}},
    "answer": "[the correct choices(A-D)]",
}}

### Correct Answer:
{len(frame_obj_dict[frame][obj_class])}
"""
                response = api_response_textonly(client=client, model=model, prompt=prompt)
                response = response.replace("```json", "")
                response = response.replace("```", "")
                try:      
                    question_json = json.loads(response)
                    question_json["reasoning"] = None
                    question_json["source"] = video_data["video"]
                    question_json["type"] = "FOC"
                    question_json["generation_message"] = {"obj": obj_class, "time": frame_time}
                    qra_list.append(question_json)
                except json.decoder.JSONDecodeError as e:
                    print("Decode Error For:", response)
                
            obj_count = 0
            frame_time = frame / fps
            frame_description = f"Target frame located at {seconds_to_mmss(frame_time)}.\n"
            for obj_class in frame_obj_dict[frame]:
                frame_description += f"The frame has {len(frame_obj_dict[frame][obj_class])} {obj_class} :\n"
                temp_count = 0
                for target_obj in frame_obj_dict[frame][obj_class]:
                    obj_count += 1
                    temp_count += 1
                    obj_data = frame_obj_dict[frame][obj_class][target_obj]
                    obj_bbox = obj_data["bbox"][ obj_data["current_index"] ]
                    frame_description += f"{obj_class}#{temp_count} was located at {obj_bbox}."
                    if len(obj_data["attributes"]):
                        frame_description += f" {obj_class}#{temp_count} have following attributes:"
                        for obj_attribute in obj_data["attributes"]:
                            frame_description += f" [{obj_attribute['attribute_name']} {obj_attribute['attribute_value']}]"
                        frame_description += '.'

                    if len(obj_data["actions"]):
                        obj_NL_action = ' '
                        for obj_action in obj_data["actions"]:
                            if frame_time >= obj_action["start_time"] and frame_time <= obj_action["end_time"] and (obj_action['action_description'] not in obj_NL_action):
                                obj_NL_action += obj_action["action_description"]
                        frame_description += obj_NL_action
                    frame_description += '\n'

            if obj_count >= 3: # this frame have more than 3 object.
                for obj_class in frame_obj_dict[frame]:
                    prompt = f"""You are an educational assistant specialized in designing problems and providing solutions. \
Given object description of target_frame, your task is to design a multiple-choice question to challenge the test-taker's perception and grounding ability.

### Attention:

The output should be in a JSON file as a Python dictionary followed output format below:
1. The question stem must contain a {obj_class} in frame description with correct and distinguishable description. \
Such as "Choose the correct bbox of the cat drinking water at 00:18."
2. The correct choice should be the corresponding bbox of target object in frame description.
3. Distractor Options (A/B/C/D except correct one) can either be other bbox in frame description, or bbox visually similar to the correct bbox, but definitely wrong. Such as a bbox derived from the true answer but with offsets more than 50 pixels on its coordinates, making them different with the true answer in position or shape.

### Output Format:
{{
    "question": "Choose the correct bbox of [target {obj_class} and description] at {seconds_to_mmss(frame_time)}",
    "choices": {{"A": "...", "B": "...", "C": "...", "D": ...}},
    "answer": "[the correct choices(A-D)]",
}}
### Frame description:
{frame_description}
"""                
                    response = api_response_textonly(client=client, model=model, prompt=prompt)
                    response = response.replace("```json", "")
                    response = response.replace("```", "")
                    try:      
                        question_json = json.loads(response)
                        question_json["reasoning"] = None
                        question_json["source"] = video_data["video"]
                        question_json["type"] = "FOS"
                        question_json["generation_message"] = {"time": frame_time}
                        qra_list.append(question_json)
                    except json.decoder.JSONDecodeError as e:
                        print("Decode Error For:", response)        
            # continue

        for key in object_dict: # iteration on each class of object
            if len(object_dict[key]) > 1:
                annotation = {}
                citation = {}
                frame_count_dict = {}
                minimum_count = 0
                count = 0

                # Iteration on Segment Object
                for distinct_index in object_dict[key]: # iteration on each object of target class
                    count += 1
                    annotation[f"{key}#{count}"] = []
                    citation[f"{key}#{count}"] = distinct_index
                    for obj_index in distinct_id_dict[distinct_index]:
                        time_and_position = {}

                        # 每一帧中出现的肯定是不重叠的(出于抑制幻觉，仅对其出现的第一帧予以注册)
                        time_0 = segment["objects"][obj_index]["time"][0]
                        if round(time_0 * fps) in frame_count_dict:
                            frame_count_dict[ round(time_0 * fps) ] += 1
                        else:
                            frame_count_dict[ round(time_0 * fps) ] = 1


                        for time_ in segment["objects"][obj_index]["time"]:                            
                            time_and_position[seconds_to_mmss(time_)] = f"{search_direction(video_data= video_data, obj_index=distinct_index, target_frame = round(time_ * fps))} video"
                        filtered_attributes = []
                        for obj_attribute in segment["objects"][obj_index]["attributes"]:
                            if obj_attribute["attachmant"] == -1:
                                attr_name = obj_attribute["attribute_name"]
                                attr_value = obj_attribute["attribute_value"]
                                filtered_attributes.append(f"{{'name':{attr_name}, 'value':{attr_value} }}")                        
                        new_dict = {
                            "time_and_position": time_and_position,
                            "attributes": filtered_attributes
                        }
                        annotation[f"{key}#{count}"].append(new_dict)
                
                for keyframe in frame_count_dict:
                    if frame_count_dict[keyframe] > minimum_count:
                        minimum_count = frame_count_dict[keyframe]

                if minimum_count == len(object_dict[key]): #无需审查
                    response_json = {}
                    for i in range(minimum_count):
                        response_json[f"distinct_{key}#{i+1}"] = [f"{key}#{i+1}"]
                else:
                    prompt = f"""You are a data analysis assistant specialized in analyzing and categorizing object records from videos. \
Given time intervals and feature records about multiple objects, determine how many distinct objects are present in the video and group the records that belong to the same object, considering that records from different categories may refer to the same object, and there are at least {minimum_count} distinct objects. \
The output should be in a JSON file as a Python dictionary, where the keys represent each unique object (e.g., "distinct_{key}#1") and the values are all the record categories representing that object (e.g., ["{key}#1", "{key}#2"]).'

### Output Format:
1. Your output should be formed in a JSON file.
2. Only provide the Python dictionary string.

EXAMPLE:
{{
    "distinct_{key}#1": ["{key}#1", "{key}#2"],
    "distinct_{key}#2": ["{key}#3"],
    "distinct_{key}#3": ["{key}#4”]
}}

### Object Record:
{annotation}
"""   
                    response = api_response_textonly(client=client, model=model, prompt=prompt)
                    response = response.replace("```json", "")
                    response = response.replace("```", "")
                    try:
                        response_json = json.loads(response)
                    except json.decoder.JSONDecodeError as e:
                        print("Decode Error For:", response)
                        continue
                
                # print(response_json)

                raw_obj_citation = {}
                raw_obj_annotation = {}
                raw_obj_action = {}
                for distinct_obj in response_json:

                    raw_obj_citation[distinct_obj] = []
                    raw_obj_annotation[distinct_obj] = []
                    raw_obj_action[distinct_obj] = []

                    for distinct_obj_id in response_json[distinct_obj]:
                        origial_did = citation[distinct_obj_id]
                        raw_obj_citation[distinct_obj].append(origial_did)
                        raw_obj_action[distinct_obj].extend(object_action_dict[origial_did])
                        raw_obj_annotation[distinct_obj].extend(annotation[distinct_obj_id])


                segment_obj_citation[key] = raw_obj_citation
                segment_obj_annotation[key] = raw_obj_annotation
                segment_obj_action[key] = raw_obj_action
            else:
                distinct_id = object_dict[key][0]
                segment_obj_citation[key] = [{f"distinct_{key}#1": distinct_id}]
                segment_obj_annotation[key] = {f"distinct_{key}#1": []}
                segment_obj_action[key] = {f"distinct_{key}#1": object_action_dict[distinct_id]}
                for obj_index in distinct_id_dict[distinct_id]:
                    time_and_position = {}
                    for time_ in segment["objects"][obj_index]["time"]:                            
                        time_and_position[seconds_to_mmss(time_)] = f"{search_direction(video_data= video_data, obj_index=distinct_id, target_frame = round(time_ * fps))} video"
                    filtered_attributes = []
                    for obj_attribute in segment["objects"][obj_index]["attributes"]:
                        if obj_attribute["attachmant"] == -1:
                            attr_name = obj_attribute["attribute_name"]
                            attr_value = obj_attribute["attribute_value"]
                            filtered_attributes.append(f"{{'name':{attr_name}, 'value':{attr_value} }}")                        
                    new_dict = {
                        "time_and_position": time_and_position,
                        "attributes": filtered_attributes
                    }
                    segment_obj_annotation[key][f"distinct_{key}#1"].append(new_dict)

        # segment_obj_annotation: 物品特征, segment_obj_annotation - key - distinct_{key}#{No.}
        # segment_obj_action: 物品动作, segment_obj_annotation - key - distinct_{key}#{No.}

        if seg_total_time > 3:
            for key in segment_obj_citation:
                if len(segment_obj_citation[key]) > 1:
                    prompt = f"""You are an educational assistant specialized in designing problems and providing solutions and reasoning. \
Given time intervals and feature records about multiple objects, a short-answer question and its correct answer, \
your task is to transform it into a multiple-choice question format and generate the problem-solving process step by step.

### Attention:
1. Your reasoning step should be no more than {min(len(segment_obj_citation[key]) + 2, 6)} steps.
2. Your reasoning process must not directly use labels in the object record(eg. distinct_{key}#1), instead you should guide students to find relevant information for the answer from the original video.
3. Your reasoning process must contain the overall feature of each distinct object and its first time appears in the video.
4. The last step of your reasoning process must end with "so the correct choice is ...".

The output should be in a JSON file as a Python dictionary followed output format below:

### Output Format:
{{
    "question": "How many distinct {key} appears in the video from {seg_start_time} to {seg_end_time}?",
    "answer": "[The correct choice]",
    "choices": {{"A": "...", "B": "...", "C": "...", D: "..."}},
    "reasoning": {{
            "step1": "[Your Reasoning]",
            "step2": "[Your Reasoning]", 
            ...
            }}
}}

### Object Record:
{segment_obj_annotation[key]}

### Correct Answer
{len(segment_obj_citation[key])}
"""                  
                    response = api_response_textonly(client=client, model=model, prompt=prompt)
                    response = response.replace("```json", "")
                    response = response.replace("```", "")
                    try:        
                        question_json = json.loads(response)
                        question_json["source"] = video_data["video"]
                        question_json["type"] = "SOC" 
                        question_json["generation_message"] = {"obj": key, "time": [seg_start_time, seg_end_time]}
                        qra_list.append(question_json)
                    except json.decoder.JSONDecodeError as e:
                        print("Decode Error For:", response)
        
        # Generate Combination Features
        segment_action_count = {}
        segment_combination_feature = {}
        for key in segment_obj_annotation: # iteration on each class of object
            segment_action_count[key] = 0
            segment_combination_feature[key] = {}
            for distinct_key in segment_obj_annotation[key]: # iteration on each distinct object of target class
                segment_combination_feature[key][distinct_key] = {
                    "feature": segment_obj_annotation[key][distinct_key],
                    "action": segment_obj_action[key][distinct_key]
                }
                segment_action_count[key] += len(segment_obj_action[key][distinct_key])
        
        for key in segment_action_count: # iteration on each class of object
            # Segment Event Detection 
            if segment_action_count[key] > 6: # have massive events, quest which event didn't happen to target object
                prompt = f"""You are an educational assistant specialized in designing problems and providing solutions and reasoning. \
Given event and feature records about multiple objects from {seg_start_time} to {seg_end_time} of video, \
your task is to design a multiple-choice question to challenge the test-taker's memory ability.

### Attention:
1. Your question should focus on a specific {key} and events happened to it.
2. There may be multiple {key}, so the question stem must reflect the unique feature of the target one.
3. The question stem must follow the shape like "Between {seg_start_time} and {seg_end_time}, which event didn't happen to [the {key} with description]?".
4. Different choices of the question should be distinguishable. Several alternative wrong choices include fictional event and event happened to other object.

The output should be in a JSON file as a Python dictionary followed output format below:

### Output Format:
{{
    "question": "Between {seg_start_time} and {seg_end_time}, which event didn't happen to [the {key} with description]?",
    "answer": "[The correct choice(A-D)]", 
    "choices": {{"A": "...", "B": "...", "C": "...", D: "..."}}
}}

### Event and feature record:
{segment_combination_feature}
"""        
                response = api_response_textonly(client=client, model=model, prompt=prompt)
                response = response.replace("```json", "")
                response = response.replace("```", "")
                try:    
                    question_json = json.loads(response)
                    question_json["reasoning"] = None
                    question_json["source"] = video_data["video"]
                    question_json["type"] = "SED"
                    question_json["generation_message"] = {"obj": key, "time": [seg_start_time, seg_end_time]} 
                    qra_list.append(question_json)
                except json.decoder.JSONDecodeError as e:
                    print("Decode Error For:", response)
                # print(question_json)
                 
            # Segment Event Detection
            elif segment_action_count[key] > 0: # Limited events, quest which event happen to target object
                prompt = f"""You are an educational assistant specialized in designing problems and providing solutions and reasoning. \
Given event and feature records about multiple objects from {seg_start_time} to {seg_end_time} of video, \
your task is to design a multiple-choice question to challenge the test-taker's memory ability.

### Attention:
1. Your question should focus on a specific {key} and events happened to it.
2. There may be multiple {key}, so the question stem must reflect the unique feature of the target one.
3. The question stem must follow the shape like "Between {seg_start_time} and {seg_end_time}, which event happened to [the {key} with description]?".
4. Different choices of the question should be distinguishable. Several alternative wrong choices include fictional event and event happened to other object.

The output should be in a JSON file as a Python dictionary followed output format below:

### Output Format:
{{
    "question": "Between {seg_start_time} and {seg_end_time}, which event happened to [the {key} with description]?",
    "answer": "[The correct choice(A-D)]", 
    "choices": {{"A": "...", "B": "...", "C": "...", D: "..."}}
}}

### Event and feature record:
{segment_combination_feature}
"""        
                response = api_response_textonly(client=client, model=model, prompt=prompt)
                response = response.replace("```json", "")
                response = response.replace("```", "")
                try:    
                    question_json = json.loads(response)
                    question_json["reasoning"] = None
                    question_json["source"] = video_data["video"]
                    question_json["type"] = "SED" 
                    question_json["generation_message"] = {"obj": key, "time": [seg_start_time, seg_end_time]}
                    qra_list.append(question_json)
                except json.decoder.JSONDecodeError as e:
                    print("Decode Error For:", response)
                # print(question_json)

        for key in segment_action_count: # iteration on each class of object
            # Segment Anomaly Detection
            if segment_action_count[key] > 2:
                prompt = f"""You are an educational assistant specialized in designing problems and providing solutions and reasoning. \
Given event and feature records about multiple objects from {seg_start_time} to {seg_end_time} of video, \
your task is to design a multiple-choice question to challenge the test-taker's memory ability.

### Attention:
1. Your question should focus on events happened in records.
2. The question stem must follow the shape like "Between {seg_start_time} and {seg_end_time}, which event didn't happen?".
3. Different choices of the question should be distinguishable. The answer choice must be a fictional event happen to a {key} which is not in record. 
4. Other choices should be events happened to various kinds of objects. Avoid designing choices with events only happened to {key}.

The output should be in a JSON file as a Python dictionary followed output format below:

### Output Format:
{{
    "question": "Between {seg_start_time} and {seg_end_time}, which event didn't happen?",
    "answer": "[The correct choice(A-D)]", 
    "choices": {{"A": "...", "B": "...", "C": "...", D: "..."}}
}}

### Event and feature record:
{segment_combination_feature}
"""                   
                response = api_response_textonly(client=client, model=model, prompt=prompt)
                response = response.replace("```json", "")
                response = response.replace("```", "")
                try:    
                    question_json = json.loads(response)
                    question_json["reasoning"] = None
                    question_json["source"] = video_data["video"]
                    question_json["generation_message"] = {"obj": key, "time": [seg_start_time, seg_end_time]}
                    question_json["type"] = "SAD" 
                    qra_list.append(question_json)
                except json.decoder.JSONDecodeError as e:
                    print("Decode Error For:", response)
                # qra_list.append(question_json)

        # generate frame-class question

    return qra_list

def generate_qa_via_multihop_construction(video_data, client, model):
    qra_list = []
    video_stsg = video_data['video_graph']
    fps = video_data["fps"]
    for segment in video_stsg['segments']:
        for tracklet in segment['tracklets']:
            for tracklet_head, tracklet_tail in combinations(tracklet['objects'], 2):
                available_prefix = []
                # tracklet_head = tracklet['objects'][0]
                # tracklet_tail = tracklet['objects'][-1]

                start_time = segment['objects'][tracklet_head]['time'][0]
                end_time = segment['objects'][tracklet_tail]['time'][-1]
                raw_answer = segment['objects'][tracklet_tail]['bbox'][-1]

                # 时间太短则没有记忆能力
                if end_time-start_time < 1.5:
                    continue
                elif end_time-start_time > 30 and random.random() > 0.5:
                    continue

                #向头部延展:
                for relation in segment['relations']:
                    if relation['object'] == tracklet_head and round(start_time * fps) >= round(relation['time'][0] * fps) and round(start_time * fps) <= round(relation['time'][-1] * fps) and relation['subject'] not in tracklet['objects']:
                        available_prefix.append(relation)
                for action in segment['actions']:
                    if action['subject'] == tracklet_head and round(start_time * fps) >= round(action['start_time'] * fps) and round(start_time * fps) <= round(action['end_time'] * fps):
                        available_prefix.append(action)
                
                for prefix in available_prefix:
                    raw_question = ''
                    raw_choice = random.choice(['A', 'B', 'C', 'D'])
                    if 'relation_id' in prefix:
                        prefix_obj = segment['objects'][prefix['subject']]
                        target_obj = segment['objects'][prefix['object']]

                        raw_question += f""" For the {prefix_obj['object_name']} with following features: {get_obj_feature(video_data, prefix_obj['distinct_label'], round(start_time*fps))}, \
at {seconds_to_mmss(start_time)}, it (is) {prefix['predicate']} another object. Where is the object {round(end_time - start_time, 2)} seconds after the {prefix_obj['object_name']} {prefix['predicate']} it?
"""
                        raw_reasoning = {
                            "step1": f"""At {seconds_to_mmss(start_time)}, the {target_obj['object_name']} at {target_obj['bbox'][0]} satisfy the description "{prefix_obj['object_name']} {prefix['predicate']} {target_obj['object_name']}". So we need to find where is the {target_obj['object_name']} at {round(end_time - start_time, 2)} seconds later.""",
                            "step2": f"""{round(end_time - start_time, 2)} seconds later is {seconds_to_mmss(end_time)}. At {seconds_to_mmss(end_time)}, the {target_obj['object_name']} located at {raw_answer}, which matches choice {raw_choice} best."""
                        }
                        prompt = f"""You are an educational assistant specialized in designing problems, given question stem with structured data, reasoning step and correct answer, your task is to rewrite it to a multiple-choice question with natural-language style question stem and reasoning step.
### Attention:
1. The last step of your reasoning process must end with "so the correct choice is ...".
2. Distractor Options (A/B/C/D except correct one): Each must be scene-relevant, visually similar to the correct bbox, but definitely wrong. They can be true answer with offsets (usually more than 50 pixels) on its coordinates, making them different with the true answer in position or shape. They can also be bounding box relevant to the original positon {target_obj['bbox'][0]}.
3. The description of {prefix_obj['object_name']} in the question should be concise and accurate.
4. The choice of {raw_choice} should be {raw_answer} or its its round-to-the-nearest-ten form.
                
The output should be in a JSON file as a Python dictionary followed output format below:

### Output Format:
{{
    "question": "For the [{prefix_obj['object_name']} with natural language description] at {seconds_to_mmss(start_time)}, [description about the relation between {prefix_obj['object_name']} and another object/person]. Where is the object/person {round(end_time - start_time, 2)} seconds after ...?",
    "answer": "{raw_choice}",
    "choices": {{"A": "...", "B": "...", "C": "...", D: "..."}},
    "reasoning": {{
            "step1": "[Your Reasoning]",
            "step2": "[Your Reasoning]", 
            ...
            }}
}}

### Raw Question:
{raw_question}

### Raw Answer:
{raw_answer}

### Raw Reasoning:
{raw_reasoning}
"""
                        response = api_response_textonly(client=client, model=model, prompt=prompt)
                        response = response.replace("```json", "")
                        response = response.replace("```", "")
                        try:       
                            question_json = json.loads(response)
                            question_json["source"] = video_data["video"]
                            question_json["type"] = "MHR"
                            question_json["generation_message"] = {"obj": [prefix_obj['object_name'], target_obj['object_name']], "time": [start_time, end_time], "bbox":[target_obj['bbox'][0], raw_answer]}
                            qra_list.append(question_json)
                        except json.decoder.JSONDecodeError as e:
                            print("Decode Error For:", response)

                    if 'action_id' in prefix:
                        action_description = ''
                        target_obj = segment['objects'][prefix['subject']]
                        if 'None' in prefix['object']:
                            action_description = prefix['predicate']
                        elif isinstance(prefix['object'], int):
                            action_description = f"{prefix['predicate']} {segment['objects'][prefix['object']]['object_name']}"
                        else:
                            action_description = f"{prefix['predicate']} {prefix['object']}"
                        
                        start_bbox = search_bbox(video_data, target_obj['distinct_label'], round(start_time * fps))
                        if start_bbox is None:
                            continue
                        raw_question += f"For the object {action_description} at {seconds_to_mmss(start_time)}. Where is the object {round(end_time - start_time, 2)} seconds later?"
                        raw_reasoning = {
                            "step1": f"At {seconds_to_mmss(start_time)}, the {target_obj['object_name']} located at {start_bbox} satisfy the description '{action_description}'. So we need to find where is the {target_obj['object_name']} at {round(end_time - start_time, 2)} seconds later.",
                            "step2": f"""{round(end_time - start_time, 2)} seconds later is {seconds_to_mmss(end_time)}. At {seconds_to_mmss(end_time)}, the {target_obj['object_name']} located at {raw_answer}, which matches choice {raw_choice} best."""
                        }                    
                        prompt = f"""You are an educational assistant specialized in designing problems, given question stem with structured data, reasoning step and correct answer, your task is to rewrite it to a multiple-choice question with natural-language style question stem and reasoning step.

### Attention:
1. The last step of your reasoning process must end with "so the correct choice is ...".
2. Distractor Options (A/B/C/D except correct one): Each must be scene-relevant, visually similar to the correct bbox, but definitely wrong. They can be true answer with offsets (usually more than 50 pixels) on its coordinates, making them different with the true answer in position or shape. They can also be bounding box relevant to the original positon {start_bbox}.
3. The action description in the question should be concise and accurate.
4. The choice of {raw_choice} should be {raw_answer} or its its round-to-the-nearest-ten form.
                
The output should be in a JSON file as a Python dictionary followed output format below:

### Output Format:
{{
    "question": "For the [object with action description] at {seconds_to_mmss(start_time)}. Where is the object {round(end_time - start_time, 2)} seconds later?",
    "answer": "{raw_choice}",
    "choices": {{"A": "...", "B": "...", "C": "...", D: "..."}},
    "reasoning": {{
            "step1": "[Your Reasoning]",
            "step2": "[Your Reasoning]", 
            ...
            }}
}}

### Raw Question:
{raw_question}

### Raw Answer:
{raw_answer}

### Raw Reasoning:
{raw_reasoning}
"""
                        response = api_response_textonly(client=client, model=model, prompt=prompt)
                        response = response.replace("```json", "")
                        response = response.replace("```", "")
                        try:    
                            question_json = json.loads(response)
                            question_json["source"] = video_data["video"]
                            question_json["type"] = "MHA"
                            question_json["generation_message"] = {"obj": target_obj['object_name'], "time": [start_time, end_time], "bbox":[start_bbox, raw_answer]}
                            qra_list.append(question_json)
                        except json.decoder.JSONDecodeError as e:
                            print("Decode Error For:", response)
    return qra_list

def generate_qa_via_reverse_multihop_construction(video_data, client, model):
    ### 倒序的生成推理问题
    
    qra_list = []
    video_stsg = video_data['video_graph']
    fps = video_data["fps"]
    for segment in video_stsg['segments']:
        for tracklet in segment['tracklets']:
            for tracklet_head, tracklet_tail in combinations(tracklet['objects'], 2):
                available_prefix = []
                # tracklet_head = tracklet['objects'][0]
                # tracklet_tail = tracklet['objects'][-1]

                start_time = segment['objects'][tracklet_head]['time'][0]
                end_time = segment['objects'][tracklet_tail]['time'][-1]

                raw_answer = segment['objects'][tracklet_head]['bbox'][0]

                # 时间太短则没有记忆能力
                if end_time-start_time < 1.5:
                    continue
                elif end_time-start_time > 30 and random.random() > 0.5:
                    continue

                #向头部延展:
                for relation in segment['relations']:
                    if relation['object'] == tracklet_tail and round(end_time * fps) >= round(relation['time'][0] * fps) and round(end_time * fps) <= round(relation['time'][-1] * fps) and relation['subject'] not in tracklet['objects']:
                        available_prefix.append(relation)
                for action in segment['actions']:
                    if action['subject'] == tracklet_tail and round(end_time * fps) >= round(action['start_time'] * fps) and round(end_time * fps) <= round(action['end_time'] * fps):
                        available_prefix.append(action)
                
                for prefix in available_prefix:
                    raw_question = ''
                    raw_choice = random.choice(['A', 'B', 'C', 'D'])
                    if 'relation_id' in prefix:
                        prefix_obj = segment['objects'][prefix['subject']]
                        target_obj = segment['objects'][prefix['object']]

                        raw_question += f""" For the {prefix_obj['object_name']} with following features: {get_obj_feature(video_data, prefix_obj['distinct_label'], round(end_time*fps))}, \
at {seconds_to_mmss(end_time)}, it (is) {prefix['predicate']} another object. Where is the object {round(end_time - start_time, 2)} seconds before the {prefix_obj['object_name']} {prefix['predicate']} it?
"""
                        raw_reasoning = {
                            "step1": f"""At {seconds_to_mmss(end_time)}, the {target_obj['object_name']} at {target_obj['bbox'][-1]} satisfy the description "{prefix_obj['object_name']} {prefix['predicate']} {target_obj['object_name']}". So we need to find where is the {target_obj['object_name']} at {round(end_time - start_time, 2)} seconds before.""",
                            "step2": f"""{round(end_time - start_time, 2)} seconds before is {seconds_to_mmss(start_time)}. At {seconds_to_mmss(start_time)}, the {target_obj['object_name']} located at {raw_answer}, which matches choice {raw_choice} best."""
                        }
                        prompt = f"""You are an educational assistant specialized in designing problems, given question stem with structured data, reasoning step and correct answer, your task is to rewrite it to a multiple-choice question with natural-language style question stem and reasoning step.
### Attention:
1. The last step of your reasoning process must end with "so the correct choice is ...".
2. Distractor Options (A/B/C/D except correct one): Each must be scene-relevant, visually similar to the correct bbox, but definitely wrong. They can be true answer with offsets (usually more than 50 pixels) on its coordinates, making them different with the true answer in position or shape. They can also be bounding box relevant to the original positon {target_obj['bbox'][0]}.
3. The description of {prefix_obj['object_name']} in the question should be concise and accurate.
4. The choice of {raw_choice} should be {raw_answer} or its its round-to-the-nearest-ten form.
                
The output should be in a JSON file as a Python dictionary followed output format below:

### Output Format:
{{
    "question": "For the [{prefix_obj['object_name']} with natural language description] at {seconds_to_mmss(end_time)}, [description about the relation between {prefix_obj['object_name']} and another object/person]. Where is the object/person {round(end_time - start_time, 2)} seconds before ...?",
    "answer": "{raw_choice}",
    "choices": {{"A": "...", "B": "...", "C": "...", D: "..."}},
    "reasoning": {{
            "step1": "[Your Reasoning]",
            "step2": "[Your Reasoning]", 
            ...
            }}
}}

### Raw Question:
{raw_question}

### Raw Answer:
{raw_answer}

### Raw Reasoning:
{raw_reasoning}
"""
                        response = api_response_textonly(client=client, model=model, prompt=prompt)
                        response = response.replace("```json", "")
                        response = response.replace("```", "")
                        try:       
                            question_json = json.loads(response)
                            question_json["source"] = video_data["video"]
                            question_json["type"] = "RMHR"
                            question_json["generation_message"] = {"obj": [prefix_obj['object_name'], target_obj['object_name']], "time": [start_time, end_time], "bbox":[raw_answer, target_obj['bbox'][-1]]}
                            qra_list.append(question_json)
                        except json.decoder.JSONDecodeError as e:
                            print("Decode Error For:", response)

                    if 'action_id' in prefix:
                        action_description = ''
                        target_obj = segment['objects'][prefix['subject']]
                        if 'None' in prefix['object']:
                            action_description = prefix['predicate']
                        elif isinstance(prefix['object'], int):
                            action_description = f"{prefix['predicate']} {segment['objects'][prefix['object']]['object_name']}"
                        else:
                            action_description = f"{prefix['predicate']} {prefix['object']}"
                        
                        start_bbox = search_bbox(video_data, target_obj['distinct_label'], round(end_time * fps))
                        if start_bbox is None:
                            continue
                        raw_question += f"For the object {action_description} at {seconds_to_mmss(end_time)}. Where is the object {round(end_time - start_time, 2)} seconds before?"
                        raw_reasoning = {
                            "step1": f"At {seconds_to_mmss(end_time)}, the {target_obj['object_name']} located at {start_bbox} satisfy the description '{action_description}'. So we need to find where is the {target_obj['object_name']} at {round(end_time - start_time, 2)} seconds before.",
                            "step2": f"""{round(end_time - start_time, 2)} seconds before is {seconds_to_mmss(start_time)}. At {seconds_to_mmss(start_time)}, the {target_obj['object_name']} located at {raw_answer}, which matches choice {raw_choice} best."""
                        }                    
                        prompt = f"""You are an educational assistant specialized in designing problems, given question stem with structured data, reasoning step and correct answer, your task is to rewrite it to a multiple-choice question with natural-language style question stem and reasoning step.

### Attention:
1. The last step of your reasoning process must end with "so the correct choice is ...".
2. Distractor Options (A/B/C/D except correct one): Each must be scene-relevant, visually similar to the correct bbox, but definitely wrong. They can be true answer with offsets (usually more than 50 pixels) on its coordinates, making them different with the true answer in position or shape. They can also be bounding box relevant to the original positon {start_bbox}.
3. The action description in the question should be concise and accurate.
4. The choice of {raw_choice} should be {raw_answer} or its its round-to-the-nearest-ten form.
                
The output should be in a JSON file as a Python dictionary followed output format below:

### Output Format:
{{
    "question": "For the [object with action description] at {seconds_to_mmss(end_time)}. Where is the object {round(end_time - start_time, 2)} seconds before?",
    "answer": "{raw_choice}",
    "choices": {{"A": "...", "B": "...", "C": "...", D: "..."}},
    "reasoning": {{
            "step1": "[Your Reasoning]",
            "step2": "[Your Reasoning]", 
            ...
            }}
}}

### Raw Question:
{raw_question}

### Raw Answer:
{raw_answer}

### Raw Reasoning:
{raw_reasoning}
"""
                        response = api_response_textonly(client=client, model=model, prompt=prompt)
                        response = response.replace("```json", "")
                        response = response.replace("```", "")
                        try:    
                            question_json = json.loads(response)
                            question_json["source"] = video_data["video"]
                            question_json["type"] = "RMHA"
                            question_json["generation_message"] = {"obj": target_obj['object_name'], "time": [start_time, end_time], "bbox":[raw_answer, start_bbox]}
                            qra_list.append(question_json)
                        except json.decoder.JSONDecodeError as e:
                            print("Decode Error For:", response)
    return qra_list