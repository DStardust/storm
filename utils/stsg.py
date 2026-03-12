import copy
import re

class Attribute:
    def __init__(self, attribute_name, attribute_value, attachment=-1):
        self.attribute_name = attribute_name
        self.attribute_value = attribute_value
        self.attachment = attachment
    
    def __str__(self):
        return f"Attribute({self.attribute_name}, {self.attribute_value}, {self.attachment})"
    
    def __repr__(self):
        return self.__str__()
    
    def to_dict(self):
        return {
            "attribute_name": self.attribute_name,
            "attribute_value": self.attribute_value,
            "attachmant": self.attachment,
        }

def parsing_Attribute(dict_Attribute):
    attribute = Attribute(
        attribute_name= dict_Attribute["attribute_name"],
        attribute_value= dict_Attribute["attribute_value"],
        attachment= dict_Attribute["attachmant"]
    )
    return attribute



class Object:
    def __init__(self, segment_id, frame_id, object_id, object_name, distinct_label, time, bbox):
        self.segment_id = segment_id
        self.frame_id = frame_id
        self.object_id = object_id
        self.object_name = object_name
        self.distinct_label = distinct_label
        self.bbox = [bbox]
        self.attributes = []
        self.actions = []
        self.time = [round(time,4)]
        self.object_type = "static"
        self.relations = []
    
    def add_subject_relation(self, relation):
        new_relation = copy.deepcopy(relation)
        new_relation.subject = -1
        self.relations.append(new_relation)
    
    def add_object_relation(self, relation):
        new_relation = copy.deepcopy(relation)
        new_relation.object = -1
        self.relations.append(new_relation)

    def add_action(self, action):
        self.actions.append(action)

    def add_attribute(self, attribute):
        self.attributes.append(attribute)
    
    def add_location(self, location):
        self.location = location
    
    def add_action(self, action):
        if self.object_type == "static":
            self.object_type = "dynamic"
        self.actions.append(action)
    
    def get_correspond_attributes_id(self, subject_str):
        for i in range(0, len(self.attributes)):
            if self.attributes[i].attribute_value == subject_str:
                return i
        return -1

    def __str__(self):
        return f"Object({self.segment_id}, {self.frame_id},  {self.object_id}, {self.object_name}, {self.distinct_label}, {self.attributes}, {self.relations}, {self.actions}, {self.time}, {self.bbox})"
    
    def __repr__(self):
        return self.__str__()
    
    def to_dict(self):
        return {
            "segment_id": self.segment_id,
            "frame_id": self.frame_id,
            "object_id": self.object_id,
            "object_name": self.object_name,
            "distinct_label": self.distinct_label,
            "attributes": [attr.to_dict() for attr in self.attributes],
            "actions": [action.to_dict() for action in self.actions],
            # "actions": [action.action_description for action in self.actions],
            "object_type": self.object_type,
            "time": [time_value for time_value in self.time], #self.time,
            "bbox": [bbox_value for bbox_value in self.bbox], #self.bbox,
        }

def parsing_Object(dict_object):

    new_object = Object(
        segment_id= dict_object["segment_id"],
        frame_id= dict_object["frame_id"],
        object_id= dict_object["object_id"],
        object_name= dict_object["object_name"],
        distinct_label= dict_object["distinct_label"],
        time= dict_object["time"][0],
        bbox= dict_object["bbox"][0]
    )
    for dict_arrtibute in dict_object["attributes"]:
        new_object.add_attribute(parsing_Attribute(dict_arrtibute))

    for dict_action in dict_object["actions"]:
        # new_object.add_action(dict_action)
        new_object.add_action(parsing_Action(dict_action))

    for idx in range(1, len(dict_object["time"])):
        new_object.time.append(dict_object["time"][idx])
    
    for idx in range(1, len(dict_object["bbox"])):
        new_object.bbox.append(dict_object["bbox"][idx])

    return new_object

class Relationship:
    def __init__(self,segment_id, frame_id, relation_id,subject,predicate, object ,time):
        self.segment_id = segment_id
        self.frame_id = frame_id
        self.relation_id = relation_id
        self.subject = subject
        self.predicate = predicate
        self.object = object
        self.time = [round(time,4)]

    def __str__(self):
        return f"Relationship({self.segment_id}, {self.frame_id}, {self.relation_id}, {self.subject}, {self.predicate}, {self.object}, {self.time})"

    def __repr__(self):
        return self.__str__()

    def to_dict(self):
        return {
            "segment_id": self.segment_id,
            "frame_id": self.frame_id,
            "relation_id": self.relation_id,
            "subject": int(self.subject),
            "predicate": self.predicate,
            "object": int(self.object),
            "time": [time_value for time_value in self.time],
        }

def parsing_Relationship(dict_Relationship):
    new_relationship = Relationship(
        segment_id= dict_Relationship["segment_id"],
        relation_id= dict_Relationship["relation_id"],
        frame_id= dict_Relationship["frame_id"],
        subject= dict_Relationship["subject"],
        predicate= dict_Relationship["predicate"],
        object= dict_Relationship["object"],
        time= dict_Relationship["time"][0]
    )
    for idx in range(1, len(dict_Relationship["time"])):
        new_relationship.time.append(dict_Relationship["time"][idx])
    return new_relationship

# SG for a key frame
class FrameSG:
    def __init__(self, frame_id, time):
        self.frame_id = frame_id
        self.time = round(time,4)
        self.objects = []
        self.relations = []

    def add_object(self, object):
        self.objects.append(object)
    
    def add_relation(self, relation):
        self.relations.append(relation)
    
    def __str__(self):
        return f"FrameSG({self.frame_id}, {self.time}, {self.objects}, {self.relations})"
    
    def __repr__(self):
        return self.__str__()

def is_attrbute_similar(attributes1, attributes2, threshold=0.8):
    num_similar_values = 0
    num_total_values = 0
    for attr1 in attributes1:
        for attr2 in attributes2:
            if attr1.attribute_name == attr2.attribute_name:
                if attr1.attribute_value == attr2.attribute_value:
                    num_similar_values += 1
            num_total_values += 1
    
    # 计算相似度比例
    similarity_score = num_similar_values / num_total_values if num_total_values > 0 else 0
    if similarity_score >= threshold:
        return True
    return False

def combine_frame_sgs(frame_sgs):
    objects = []
    relations = []

    for frame_sg in frame_sgs:
        for obj in frame_sg.objects:
            obj_exist = False
            for o in objects:
                if o.object_name == obj.object_name:
                    if is_attrbute_similar(o.attributes, obj.attributes):
                        o.tracklet.append(obj.tracklet[0])
                        for attr in obj.attributes:
                            if attr.attribute_name not in o.attributes:
                                o.attributes.append(attr)
                        for rel in frame_sg.relations:
                            if rel.subject == obj.object_id:
                                rel.subject = o.object_id
                            if rel.object == obj.object_id:
                                rel.object = o.object_id
                        obj_exist = True
                        break
            if not obj_exist:
                objects.append(obj)

        for rel in frame_sg.relations:
            rel_exist = False
            for r in relations:
                if r.subject == rel.subject and r.object == rel.object and r.predicate == rel.predicate:
                    r.tracklet.append(rel.tracklet[0])
                    rel_exist = True
                    break
            if not rel_exist:
                relations.append(rel)                   

    return  objects, relations

class Action:
    def __init__(self, segment_id, action_id, action_description, start_time, end_time, tracklet_id, subject, predicate, object):
        self.segment_id = segment_id
        self.action_id = action_id
        self.action_description = action_description  
        self.start_time = round(start_time,2)
        self.end_time = round(end_time,2)
        self.tracklet_id = tracklet_id
        self.subject = subject
        self.predicate = predicate
        self.object = object
    
    def __str__(self):
        return f"Action({self.segment_id}, {self.action_id}, {self.action_description}, {self.start_time}, {self.end_time}, {self.tracklet_id}, {self.predicate}, {self.object})"
    
    def __repr__(self):
        return self.__str__()
    
    def to_dict(self):
        return {
            "segment_id": self.segment_id,
            "action_id": self.action_id,
            "action_description": self.action_description,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "tracklet_id": self.tracklet_id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object
        }

def parsing_Action(dict_action):
    new_action = Action(
        segment_id= dict_action["segment_id"],
        action_id= dict_action["action_id"],
        action_description= dict_action["action_description"],
        start_time= dict_action["start_time"], 
        end_time= dict_action["end_time"],
        tracklet_id= dict_action["tracklet_id"],
        subject= dict_action["subject"],
        predicate= dict_action["predicate"],
        object= dict_action["object"]
    )
    return new_action
    
class Event:
    def __init__(self, segment_id, event_description, start_time, end_time):
        self.segment_id = segment_id
        self.event_description = event_description  
        self.start_time = round(start_time,2)
        self.end_time = round(end_time,2) 
    
    def __str__(self):
        return f"Event({self.segment_id}, {self.event_description}, {self.start_time}, {self.end_time})"
    
    def __repr__(self):
        return self.__str__()
    
    def to_dict(self):
        return {
            "segment_id": self.segment_id,
            "event_description": self.event_description,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }

def parsing_Event(dict_Event):
    new_Event = Event(
        segment_id= dict_Event["segment_id"],
        event_description= dict_Event["event_description"],
        start_time= dict_Event["start_time"],
        end_time= dict_Event["end_time"]
    )

    return new_Event

class Tracklet:
    def __init__(self, segment_id, tracklet_id, obj1=None, obj2=None):
        self.segment_id = segment_id
        self.tracklet_id = tracklet_id
        self.objects = []
        if obj1 is not None:
            self.add_object(obj1)
        if obj2 is not None:
            self.add_object(obj2)

    def add_object(self, object):
        self.objects.append(object)
    
    def __str__(self):
        return f"Tracklet({self.segment_id}, {self.tracklet_id}, {self.objects})"

    def __repr__(self):
        return self.__str__()
    
    def to_dict(self):
        return {
            "segment_id": self.segment_id,
            "tracklet_id": self.tracklet_id,
            "objects": [obj for obj in self.objects]
        }

def parsing_Tracklet(dict_Tracklet):
    new_Tracklet = Tracklet(
        segment_id= dict_Tracklet["segment_id"],
        tracklet_id= dict_Tracklet["tracklet_id"]
    )
    for obj in dict_Tracklet["objects"]:
        new_Tracklet.add_object(obj)
    return new_Tracklet
    
# STSG for a segment
class SegmentSTSG:
    def __init__(self, segment_id,start_time, end_time, event, keyframes):
        self.segment_id = segment_id
        self.start_time = round(start_time,2)
        self.end_time = round(end_time,2)
        self.tracklets = []
        self.actions = []
        self.objects = []
        self.relations = []
        self.event = event
        self.keyframes = keyframes

    def __str__(self):
        return f"SegmentSTSG({self.segment_id}, {self.start_time}, {self.end_time}, {self.tracklets}, {self.objects}, {self.relations}, {self.actions}, {self.event}, {self.keyframes})"
    
    def __repr__(self):
        return self.__str__()

    def add_object(self, object):
        self.objects.append(object)
    
    def add_relation(self, relation):
        self.relations.append(relation)
    
    def add_tracklet(self, tracklet):
        self.tracklets.append(tracklet)
    
    def add_action(self, action):
        self.actions.append(action)

    def to_dict(self):
        return {
            "segment_id": self.segment_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "objects": [obj.to_dict() for obj in self.objects],
            "relations": [rel.to_dict() for rel in self.relations],
            "tracklets": [tracklet.to_dict() for tracklet in self.tracklets],
            "actions": [action.to_dict() for action in self.actions],
            "event": self.event.to_dict(),
            "keyframes": self.keyframes
        }

def parsing_SegmentSTSG(dict_SegmentSTSG):
    new_SegmentSTSG = SegmentSTSG(
        segment_id= dict_SegmentSTSG["segment_id"],
        start_time= dict_SegmentSTSG["start_time"], 
        end_time= dict_SegmentSTSG["end_time"],
        event= parsing_Event(dict_SegmentSTSG["event"]),
        keyframes = dict_SegmentSTSG["keyframes"]
    )
    for dict_obj in dict_SegmentSTSG["objects"]:
        new_SegmentSTSG.add_object(parsing_Object(dict_obj))

    for dict_rel in dict_SegmentSTSG["relations"]:
        new_SegmentSTSG.add_relation(parsing_Relationship((dict_rel)))

    for dict_tracklet in dict_SegmentSTSG["tracklets"]:
        new_SegmentSTSG.add_tracklet(parsing_Tracklet(dict_tracklet))

    for dict_action in dict_SegmentSTSG["actions"]:
        new_SegmentSTSG.add_action(parsing_Action(dict_action))

    return new_SegmentSTSG

class Reference:
    def __init__(self, reference_id, obj1, obj2):
        self.reference_id = reference_id
        self.obj1 = obj1
        self.obj2 = obj2
    
    def __str__(self):
        return f"Reference({self.reference_id}, {self.obj1}, {self.obj2})"
    
    def __repr__(self):
        return self.__str__()
    
    def to_dict(self):
        return {
            "reference_id": self.reference_id,
            "obj1": self.obj1.to_dict(),
            "obj2": self.obj2.to_dict()
        }

def parsing_Reference(dict_Reference):
    new_Reference = Reference(
        reference_id= dict_Reference["reference_id"],
        obj1= parsing_Object(dict_Reference["obj1"]),
        obj2= parsing_Object(dict_Reference["obj2"])
    )
    return new_Reference

# STSG for a video
class VideoSTSG:
    def __init__(self, video_id, start_time, end_time):
        self.video_id = video_id
        self.start_time = start_time
        self.end_time = end_time
        self.segments = []
        self.reference = []
        self.actions = []
        self.events = []
    
    def __str__(self):
        return f"VideoSTSG({self.video_id}, {self.start_time}, {self.end_time}, {self.segments}, {self.reference})"
    
    def __repr__(self):
        return self.__str__()
    
    def add_segment(self, segment):
        self.segments.append(segment)
    
    def add_reference(self, reference):
        self.reference.append(reference)
    
    def add_event(self, event):
        self.events.append(event)
    
    def add_action(self, action):
        self.actions.append(action)

    def to_dict(self):
        return {
            "video_id": self.video_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "segments": [segment.to_dict() for segment in self.segments],
            "reference": [ref.to_dict() for ref in self.reference],
            "actions": [action.to_dict() for action in self.actions],
            "events": [event.to_dict() for event in self.events]
        }

def parsing_VideoSTSG(VideoSTSG_dict):
    video_graph = VideoSTSG(
        video_id = VideoSTSG_dict["video_id"],
        start_time = VideoSTSG_dict["start_time"],
        end_time = VideoSTSG_dict["end_time"]
    )
    for segment_dict in VideoSTSG_dict["segments"]:
        video_graph.add_segment(parsing_SegmentSTSG(segment_dict))
    for dict_ref in VideoSTSG_dict["reference"]:
        video_graph.add_reference(parsing_Reference(dict_ref))
    for dict_event in VideoSTSG_dict["events"]:
        video_graph.add_event(parsing_Event(dict_event))
    for dict_action in VideoSTSG_dict["actions"]:
        video_graph.add_action(parsing_Action(dict_action))
    
    return video_graph

        

def object_exists(label,objects):
    return any(obj.object_name == label for obj in objects)

def get_object_by_label(object_name, objects):
    for obj in objects:
        if obj.object_name == object_name:
            return obj
    return None

def find_tracklet(obj,tracklets):
    for tracklet in tracklets:
        if obj in tracklet.objects:
            return tracklet
    return None

def rel_exists(rel,relations):
    return any(r.subject == rel.subject and r.object == rel.object and r.predicate == rel.predicate for r in relations)

def get_rel_by_subject_object_predicate(subject,object,predicate,relations):
    for rel in relations:
        if rel.subject == subject and rel.object == object and rel.predicate == predicate:
            return rel
    return None

def extract_triplets(input_str):
    """Extract NL Style Triplets to [{triplet 1}, {triplet 2} ...], ensuring complete-triplets"""
    pattern = r'<subject:(.*?),\s*predicate:(.*?),\s*object:(.*?)>'
    triplets = []
    if not re.finditer(pattern, input_str):
        pattern = r'<subject: (.*?),\s*predicate: (.*?),\s*object: (.*?)>'
    
    for match in re.finditer(pattern, input_str):
        subject = match.group(1).strip().lower()
        predicate = match.group(2).strip()
        obj = match.group(3).strip().replace('>', '').lower()
        if all([subject, predicate, obj]):
            triplets.append({
                'subject': subject,
                'predicate': predicate,
                'object': obj
            })
    return triplets


