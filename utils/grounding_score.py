import re
import json

def timestamp_to_seconds(ts_str):
    """将 MM:SS.ss 格式转换为秒数"""
    try:
        minutes, seconds = ts_str.split(':')
        return float(minutes) * 60 + float(seconds)
    except:
        return 0.0

def calculate_iou(box1, box2):
    """计算 BBox IoU (0-1)"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 < x1 or y2 < y1: return 0.0

    inter_area = (x2 - x1) * (y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    return inter_area / float(box1_area + box2_area - inter_area)

def calculate_time_similarity(t1, t2, threshold=5.0):
    """
    计算时间相似度 (0-1)。
    threshold: 最大容忍误差(秒)，超过此误差得分为0。
    """
    diff = abs(t1 - t2)
    if diff >= threshold:
        return 0.0
    return 1.0 - (diff / threshold)


def extract_features(text):
    """提取 BBox 和 Timestamp"""
    if isinstance(text, (dict, list)):
        # 将字典或列表转为字符串，这样正则才能处理
        text = json.dumps(text)
    
    bbox_pattern = re.compile(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]")
    bboxes = [list(map(int, m)) for m in bbox_pattern.findall(text)]
    
    time_pattern = re.compile(r"(\d{2}:\d{2}\.\d{2})")
    timestamps = [timestamp_to_seconds(t) for t in time_pattern.findall(text)]
    
    return bboxes, timestamps

def greedy_evaluate(pred_cot, answer_cot, time_tolerance=5.0):
    # 1. 提取
    pred_boxes, pred_times = extract_features(pred_cot)
    gt_boxes, gt_times = extract_features(answer_cot)
    
    results = {
        "bbox_score": 0.0,
        "time_score": 0.0, # 现在是 0-1 的分数
    }

    # 2. Bounding Box 贪心匹配 (找最大 IoU)
    if gt_boxes:
        iou_scores = []
        for gt in gt_boxes:
            max_score = 0.0
            if pred_boxes:
                max_score = max(calculate_iou(gt, pred) for pred in pred_boxes)
            iou_scores.append(max_score)
        results["bbox_score"] = sum(iou_scores) / len(iou_scores)
    
    # 3. Timestamp 贪心匹配 (找最大相似度)
    if gt_times:
        time_scores = []
        for gt in gt_times:
            max_score = 0.0
            if pred_times:
                # 遍历所有 pred 时间，找到得分最高(误差最小)的那个
                max_score = max(calculate_time_similarity(gt, pred, time_tolerance) for pred in pred_times)
            time_scores.append(max_score)
        results["time_score"] = sum(time_scores) / len(time_scores)
    
    return results