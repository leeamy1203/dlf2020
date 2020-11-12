import logging
import os
import json
import re

from src.data import DATA_DIR

logger = logging.getLogger(__name__)


def combine_openpose():
    openpose_dir = os.path.join(DATA_DIR, 'raw', 'openpose')
    output_dir = os.path.join(DATA_DIR, 'interim', '2dskeleton')
    
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for video_id in (os.listdir(openpose_dir)):
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


