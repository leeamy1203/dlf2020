import logging
import os
import json
import re

from src.data import DATA_DIR

logger = logging.getLogger(__name__)

def _gloss_starts_with(gloss, characters):
    for c in characters:
        if gloss.startswith(c):
            return True
    return False


def _grab_needed_video_ids(character_filter):
    video_ids_to_combine = []
    if character_filter is not None:
        with open(os.path.join(DATA_DIR, 'meta', 'WLASL_v0.3.json'), 'r') as f:
            meta_json = json.load(f)
        video_ids_to_combine = []
        for m in meta_json:
            if _gloss_starts_with(m["gloss"], character_filter):
                for entry in m["instances"]:
                    video_ids_to_combine.append(entry["video_id"])
    return video_ids_to_combine
    

def combine_openpose(character_filter: list):
    openpose_dir = os.path.join(DATA_DIR, 'raw', 'openpose')
    output_dir = os.path.join(DATA_DIR, 'interim', '2dskeleton')
    
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    video_ids_to_combine = _grab_needed_video_ids(character_filter)

    for video_id in (os.listdir(openpose_dir)):
        
        if character_filter is not None and video_id not in video_ids_to_combine:
            continue
            
        if video_id == 'cleaned':
            continue
        
        if os.path.isfile(os.path.join(output_dir, video_id + ".json")):
            # already ran this file
            logger.info("Already ran {}".format(video_id))
            continue
        logger.info("Looking at {}".format(video_id))
        video_dir = os.path.join(openpose_dir, video_id)
        ordered = {}
        count_frames = 0
        for frame_name in os.listdir(video_dir):
            frame_number = frame_name.split('_')[1]
            f = open(os.path.join(video_dir, frame_name))
            try:
                ordered[int(frame_number)] = json.load(f)
            except Exception as e:
                logger.error("error while grabbing = {}".format(os.path.join(video_dir, frame_name)))
                logger.error(e)
                continue
            count_frames += 1
        logger.info("\tSaving")
        with open(os.path.join(output_dir, video_id + ".json"), 'w') as output_file:
            json.dump(ordered, output_file)
            
            
def grab_frame_number(openpose_json_output_file: str):
    p = re.search(r"([^\\/]+)_(\d+)_keypoints\.json$", openpose_json_output_file)
    try:
        return int(p.group(2))
    except Exception as e:
        logger.error(openpose_json_output_file)
        return None
    
    
def create_training_json(character_filter):
    with open(os.path.join(DATA_DIR, 'meta', 'WLASL_v0.3.json'), 'r') as f:
        meta_json = json.load(f)
    
    files = list(os.listdir(os.path.join(DATA_DIR, 'interim', '2dskeleton')))
    video_ids = [f[0:-5] for f in files]
    
    for m in meta_json:
        if _gloss_starts_with(m["gloss"], character_filter):
            for entry in m["instances"]:
                if entry["video_id"] in video_ids:
                    training_json = {}
                    training_json[entry["video_id"]] = entry
                    with open(os.path.join(DATA_DIR, 'Training_Data.h.json'), 'a+') as outfile:
                        json.dump(training_json, outfile)
    

