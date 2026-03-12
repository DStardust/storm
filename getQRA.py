import json
import copy
import os
import utils.stsg as stsg
from qa_generation_utils import *

if os.getenv("API_QWEN_OMNI"):
    qwen_client = create_client(os.getenv("API_QWEN_OMNI"), "https://dashscope.aliyuncs.com/compatible-mode/v1")
else:
    print("API Client Creation Failed")
Model_dic = {
    "QWEN-OMNI": "qwen-omni-turbo-latest",  
    "QWEN-PLUS": "qwen-plus-latest",  # qwen-3-plus 25.06.24
    "QWEN-MAX": "qwen-max-latest" # fucking shit
    }


# stsg_directory = "../processed_stsg/internvid_g.json"
# write_path = "../generated_qa/internvid_qra_0817.json"

# stsg_directory = "../processed_stsg/panda_part1_12bed3652c824157bcccde92bd2db593_005000_000000.json"
# write_path = "../generated_qa/panda_qra_0817.json"

# stsg_directory = "../processed_stsg/sav.json"
# write_path = "../generated_qa/sav_qra_0817.json"

# stsg_directory = "../processed_stsg/webvid_019401_019450.json"
# write_path = "../generated_qa/webvid_qra_0817.json"



# stsg_directory = "../downloads/stsg_1015/sav_30k_40k.json"
# write_path = "../generated_qa/sav_benchmark_1025.json"

# stsg_directory = "../downloads/stsg_1015/anet_raw.json"
# write_path = "../generated_qa/anet_benchmark_1025.json"

# stsg_directory = "/home2/jiaodian/stsg/processed_stsg/_test_0925.json"
# write_path = "../generated_qa/_qa_test_0925.json"


# Tmux 7, 2025/11/14
# stsg_directory = "../downloads/stsg_1015/anet_raw.json"
# write_path = "../generated_qa_stage2/anet_benchmark_rmh_1025.json"

# Tmux 0, 2025/11/19
# stsg_directory = "../downloads/stsg_1015/sav_30k_40k.json"
# write_path = "../generated_qa_stage2/sav_train_1021_part2.json"
# ckpt_index = 75 + 251

# Tmux 0, 2025/11/21
# stsg_directory = "../downloads/stsg_1015/anet_raw.json"
# write_path = "../generated_qa_stage2/anet_train_1021_part2.json"
# ckpt_index = 75 + 166

# stsg_directory = "../downloads/stsg_1210/1210/anet_raw.json"
# write_path = "../generated_qa_stage2/anet_train_1210.json"

# stsg_directory = "../downloads/stsg_1210/1210/sav_10k_20k.json"
# write_path = "../generated_qa_stage2/sav_train_1210_part1.json"

stsg_directory = "../downloads/stsg_1210/1210/sav_20k_30k.json"
write_path = "../generated_qa_stage2/sav_train_1210_part2.json"
ckpt_index = 0

video_count = -1


with open(stsg_directory, 'r', encoding='utf-8') as file:
    content = file.read()
    decoder = json.JSONDecoder()
    data_list = []
    position = 0

    while position < len(content): 
        content = content[position:].lstrip() # 跳过空格
        position = 0  # 重置为处理后的content起始位置
        try:
            obj, pos = decoder.raw_decode(content)
            data_list.append(obj)
            position += pos  # 更新全局位置
        except json.JSONDecodeError: # 若无法解析，说明可能已经读完所有完整的JSON对象
            break

print(len(data_list))

# 验证代码
# raw_stsg = data_list[0]["video_graph"]
# video_stsg = stsg.parsing_VideoSTSG(raw_stsg)
# output_stsg_dict = {
#         "video": data_list[0]["video"],
#         "fps": data_list[0]["fps"],
#         "dimensions": data_list[0]["dimensions"],
#         "object_list": data_list[0]["object_list"],
#         "video_graph": video_stsg.to_dict()
# }

# with open("../processed_stsg/test_check.json", 'a') as f:
#     json.dump(output_stsg_dict, f)
count = 0
total_qra = 0
print(f"Total_Video_Count={len(data_list)}")
for video_data in data_list[ckpt_index:]:
    # 1.a.i Video Level Temporal Counting
    count +=1

    if video_count > 0 and count > video_count:
        break
    
    print(f"Parsing No.{ckpt_index + count} Video... of {len(data_list)}, {count} video already parsed")

    q_list = []
    output_list = []
    try:
        q_list.extend(generate_qa_via_video_object_iteration(video_data, qwen_client, Model_dic["QWEN-PLUS"]))
        print("Stage 0:", len(q_list))
    except Exception as e:
        print(f"Parsing {video_data['video']} via object iteration failed!")
    
    try:
        q_list.extend(generate_qa_via_video_action_iteration(video_data, qwen_client, Model_dic["QWEN-PLUS"]))
        print("Stage 1:", len(q_list))
    except Exception as e:
        print(f"Parsing {video_data['video']} via action iteration failed!")
    
    try:
        q_list.extend(generate_qa_via_video_segment_iteration(video_data, qwen_client, Model_dic["QWEN-PLUS"]))
        print("Stage 2:", len(q_list))
    except Exception as e:
        print(f"Parsing {video_data['video']} via segment iteration failed!")
    
    try:
        q_list.extend(generate_qa_via_multihop_construction(video_data, qwen_client, Model_dic["QWEN-PLUS"]))
        print("Stage 3:", len(q_list))
    except Exception as e:
        print(f"Parsing {video_data['video']} via multihop construction failed!")

    try:
        q_list.extend(generate_qa_via_reverse_multihop_construction(video_data, qwen_client, Model_dic["QWEN-PLUS"]))
        print("Stage 4:", len(q_list))
    except Exception as e:
        print(f"Parsing {video_data['video']} via multihop construction failed!")
    
    total_qra += len(q_list)
    print(f"total_qra == {total_qra}")

    for question in q_list:
        if 'question' in question and 'answer' in question and 'choices' in question and 'reasoning' in question:
            output_list.append(question)
        else:
            print("Corrupt question:", question)

    with open(write_path, 'a') as f:
        json.dump(output_list, f)
    