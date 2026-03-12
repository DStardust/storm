import os
from openai import OpenAI
import base64
import numpy as np

def create_client(api_key, base_url):
    client = OpenAI(api_key=api_key, base_url=base_url)
    return client

#  Base64 编码格式
def encode_video(video_path):
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode("utf-8")

def api_response_base64video(client, model, prompt, video_path):
    base64_video = encode_video(video_path)

    completion = client.chat.completions.create(
        # model="qwen-omni-turbo",
        model = model,
        messages=[
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": f"data:;base64,{base64_video}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        # 设置输出数据的模态，当前支持两种：["text","audio"]、["text"]
        modalities=["text"],
        # stream 必须设置为 True，否则会报错
        stream=True,
        stream_options={"include_usage": True},
        temperature=0,
    )
    response = ""
    for chunk in completion:
        if chunk.choices:
            if chunk.choices[0].delta.content:
                response = response + chunk.choices[0].delta.content
    return response


### Example:
# response = api_response_base64picture(client=qwen_client, model="qwen-omni-turbo", 
#                                       prompt="what happend in this picture?", base64_picture = base64_data[0])

def api_response_base64picture(client, model, prompt, base64_picture):

    completion = client.chat.completions.create(
        # model="qwen-omni-turbo",
        model = model,
        messages=[
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_picture}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        # 设置输出数据的模态，当前支持两种：["text","audio"]、["text"]
        modalities=["text"],
        # stream 必须设置为 True，否则会报错
        stream=True,
        stream_options={"include_usage": True},
        temperature=0,
    )
    response = ""
    for chunk in completion:
        if chunk.choices:
            if chunk.choices[0].delta.content:
                response = response + chunk.choices[0].delta.content
    return response

def api_response_base64piclist(client, model, prompt, base64_picture_list):

    user_content = []
    for base64_element in base64_picture_list:
        if base64_element:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_element}"},
                }
            )
        else:
            print("Invalid Frame:", base64_element)

    user_content.append({"type": "text", "text": prompt})


    completion = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
        },
        {
            "role": "user",
            "content": user_content,
        }
    ],
    modalities=["text"],
    stream=True,
    stream_options={"include_usage": True},
    temperature=0,
    )
    response = ""

    for chunk in completion:
        if chunk.choices:
            if chunk.choices[0].delta.content:
                response = response + chunk.choices[0].delta.content
    return response

def api_response_textonly(client, model, prompt, temperature = 0):
    completion = client.chat.completions.create(
    model=model,
    messages=[
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': prompt}],
    temperature= temperature,
    )
    response = completion.choices[0].message.content
    return response

def api_response_textonly_enable_thinking(client, model, prompt, thinking_budget = 800):
    completion = client.chat.completions.create(
    model=model,
    messages=[
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': prompt}],
    stream = True,
    temperature=0,
    extra_body={"enable_thinking": True, "thinking_budget": thinking_budget},
    )
    response = ""
    for chunk in completion:
        if chunk.choices:
            if chunk.choices[0].delta.content:
                response = response + chunk.choices[0].delta.content
    return response

def parsing_str(str_arg):
    str_raw = str_arg
    str_raw = str_raw.replace('\n', '')
    str_raw = str_raw.replace('\'', '\"')
    while ': ' in str_raw:
        str_raw = str_raw.replace(': ', ':')
    while ', ' in str_raw:
        str_raw = str_raw.replace(', ', ',')
    while '; ' in str_raw:
        str_raw = str_raw.replace('; ', ';')
    return str_raw