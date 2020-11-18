import logging
import os
import pickle
import json

import h5py
from torchnlp.word_to_vector import FastText
import numpy as np
from tqdm import tqdm

from src.data.importer import get_wlasl_words
from src.data import DATA_DIR
from src.data import util, skeletalModel, pose2D, pose2Dto3D, pose3D


logger = logging.getLogger(__name__)

# https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md#keypoint-ordering-in-c-python
OPENPOSE_BODY_INDX = [0, 1, 2, 3, 4, 5, 6, 7] # [nose, neck, rshoulder, relbow, rwrist, lshoulder, lelbow, lwrist]


def create_trainable() -> None:
    """
    Create trainable data
     1. create word embeddings
    """
    logger.info("Transforming words to word vectors using FAIR's FastText.")
    embeddings, words = get_word_vectors()
    logger.info("Saving embeddings and words as pickle files.")
    
    with open(os.path.join(DATA_DIR, 'interim', 'embeddings.pkl'), 'wb') as emb_file:
        pickle.dump(embeddings, emb_file)
    with open(os.path.join(DATA_DIR, 'interim', 'words.pkl'), 'wb') as word_file:
        pickle.dump(words, word_file)

def noNones(l):
  l2 = []
  for i in l:
    if not i is None:
      l2.append(i)
  return l2


def get_word_vectors() -> (np.ndarray, np.ndarray):
    """
    Goes through the meta data of WLASL and creates word embeddings for each using FastText.
    See: https://pytorchnlp.readthedocs.io/en/latest/source/torchnlp.word_to_vector.html#torchnlp.word_to_vector.FastText
    Returns an array of size (# of words=2000 x dimension of embedding=300) and an array of words

    This takes awhile because it needs to load the pretrained vectors which is about 6GB of data
    """
    logger.info("Loading FastText word vectors. This will take about 15min")
    vectors = FastText()  # loading vectors this take about 15 min
    words = get_wlasl_words()
    
    embeddings = []
    for w in words:
        embeddings.append(vectors[w].numpy())
    embeddings = np.array(embeddings)
    return embeddings, np.array(words)


def grab_desired_bodypoints(body_points):
    """
    Grab only the desired body points. We don't need lower body
    """
    points2 = []
    for i in OPENPOSE_BODY_INDX:
        points2.append(body_points[3 * i + 0]) # x
        points2.append(body_points[3 * i + 1]) # y
        points2.append(body_points[3 * i + 2]) # z
    return points2


def create_h5_for_video(video_id: str) -> np.ndarray:
    """
    Given video id convert the jsons of openpose for a video into h5
    :return: h5
    """
    video_dir = os.path.join(DATA_DIR, 'raw', 'openpose', video_id)
    openpose_files = sorted(os.listdir(video_dir))
    true_frame_num = 0
    frames = []
    for frame in openpose_files:
        frame_num = util.grab_frame_number(frame)
        if (frame_num - true_frame_num) > 5:
            logger.error(f"Openpose data skips for {frame_num - true_frame_num} frames for video {video_id}")
            true_frame_num = frame_num
        
        with open(os.path.join(video_dir, frame)) as json_data:
            data = json.load(json_data)
        
        if data is None or len(data["people"]) == 0:
            logger.error(f"No openpose for frame = {frame_num} for video = {video_id}")
            continue
        
        openpose_person = data['people'][0] # there should be only one person for our dataset
        
        body_points = grab_desired_bodypoints(openpose_person["pose_keypoints_2d"])
        left_hand_points = openpose_person["hand_left_keypoints_2d"]
        right_hand_points = openpose_person['hand_right_keypoints_2d']
        face_points = openpose_person['face_keypoints_2d']
        points = body_points + left_hand_points + right_hand_points + face_points
        
        if points is None:
            logger.error(f"No openpose for frame = {frame_num} for video = {video_id}")
            continue
        frames.append(points)
        true_frame_num += 1
    return np.asarray(frames, dtype="float32")
        

def create_h5_from_openpose():
    """
    Using 2d output from openpose. Create h5
    """
    
    openpose_dir = os.path.join(DATA_DIR, 'raw', 'openpose')
    for video_id in tqdm(os.listdir(openpose_dir)):
        h5_name = os.path.join(DATA_DIR, 'raw', 'h5', f"{video_id}.h5")
        if os.path.isfile(h5_name):
            continue
        hf = h5py.File(h5_name, 'w')
        converted_data = create_h5_for_video(video_id)
        hf.create_dataset(f"{video_id}-openpose.json", data=converted_data, dtype="float32")
        hf.close()


def create_3d_for_given_h5(h5_filename: str):
    dtype = "float32"
    structure = skeletalModel.getSkeletalModelStructure()
    noise = np.random.RandomState(1234)
    video_id = h5_filename[0:-3]
    f = h5py.File(os.path.join(DATA_DIR, 'raw', 'h5', h5_filename), "r")
    inputSequence_2D = np.array(f.get(f"{video_id}-openpose.json"))
    f.close()

    # Decomposition of the single matrix into three matrices: x, y, w (=likelihood)
    X = inputSequence_2D
    Xx = X[0:X.shape[0], 0:150:3]
    Xy = X[0:X.shape[0], 1:150:3]
    Xw = X[0:X.shape[0], 2:150:3]

    Face_Xx = X[0:X.shape[0], 150::3]
    Face_Xy = X[0:X.shape[0], 150::3]
    Face_Xw = X[0:X.shape[0], 150::3]

    # Normalization of the picture (x and y axis has the same scale)
    Xx, Xy = pose2D.normalization(Xx, Xy)
    Face_Xx, Face_Xy = pose2D.normalization(Face_Xx, Face_Xy)

    # Delete all skeletal models which have a lot of missing parts.
    Xx, Xy, Xw = pose2D.prune(Xx, Xy, Xw, (0, 1, 2, 3, 4, 5, 6, 7), 0.3, dtype)

    # Preliminary filtering: weighted linear interpolation of missing points.
    Xx, Xy, Xw = pose2D.interpolation(Xx, Xy, Xw, 0.99, dtype)
    # save("data/demo3.txt", [Xx, Xy, Xw])

    # Initial 3D pose estimation
    lines0, rootsx0, rootsy0, rootsz0, anglesx0, anglesy0, anglesz0, Yx0, Yy0, Yz0 = pose2Dto3D.initialization(
        Xx,
        Xy,
        Xw,
        structure,
        0.001,  # weight for adding noise
        noise,
        dtype
    )

    # Backpropagation-based filtering
    Yx, Yy, Yz = pose3D.backpropagationBasedFiltering(
        lines0,
        rootsx0,
        rootsy0,
        rootsz0,
        anglesx0,
        anglesy0,
        anglesz0,
        Xx,
        Xy,
        Xw,
        structure,
        dtype,
    )
    data = np.asarray(
        noNones([np.hstack((Yx, Face_Xx)), np.hstack((Yy, Face_Xy)), np.hstack((Yz, Face_Xw))]),
        dtype="float32")

    return data
    

def create_3d_from_h5():
    errors = []
    h5_dir = os.path.join(DATA_DIR, 'raw', 'h5')
    skeleton_dir = os.path.join(DATA_DIR, 'interim', '3dskeleton')
    for h5_file in tqdm(os.listdir(h5_dir)):
        video_id = h5_file[0:-3]
        try:
            output_dir = os.path.join(skeleton_dir, video_id + ".pkl")
            if os.path.isfile(output_dir):
                continue
            x, y, z = create_3d_for_given_h5(h5_file)
            with open(output_dir, 'wb') as file:
                pickle.dump([x, y, z], file)
        except Exception as e:
            logger.error(f"Could not create 3d for {video_id}")
            logger.error(e)
            errors.append(video_id)
    
    with open(os.path.join(DATA_DIR, 'logs', 'errors.pkl'), 'wb') as file:
        pickle.dump(errors, file)
        
        
def parse_important_skeleton_data(frame_data: dict):
    """
    Json is formatted [x, y, c] where c = confidence score
    https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md
    
     - body has 25 points but we only need the first 8 for upper body
     - hand has 21 points
     - face has 70 points (need to think about weighing these losses differently or removing some features
    :param frame_data:
    :return:
    """
    if frame_data['people'] is None or len(frame_data["people"]) == 0:
        logger.error("No open pose data.")
        return None
    person = frame_data['people'][0]
    
    body = []
    for i in OPENPOSE_BODY_INDX:
        body.append(person["pose_keypoints_2d"][3 * i + 0]) # x
        body.append(person["pose_keypoints_2d"][3 * i + 1]) # y
    
    left_hand = []
    for i in range(21):
        left_hand.append(person["hand_left_keypoints_2d"][3 * i + 0]) # x
        left_hand.append(person["hand_left_keypoints_2d"][3 * i + 1]) # y
    
    right_hand = []
    for i in range(21):
        right_hand.append(person["hand_right_keypoints_2d"][3 * i + 0]) # x
        right_hand.append(person["hand_right_keypoints_2d"][3 * i + 1]) # y
    
    face = []
    for i in range(70):
        face.append(person["face_keypoints_2d"][3 * i + 0]) # x
        face.append(person["face_keypoints_2d"][3 * i + 1]) # y
        
    return body + left_hand + right_hand + face
    
    
def _normalize_data(skeleton_arr: np.ndarray) -> np.ndarray:
    points = skeleton_arr.shape[-1]
    Xx = skeleton_arr[:, 0:points:2]
    Xy = skeleton_arr[:, 1:points:2]
    Xx, Xy = pose2D.normalization(Xx, Xy)
    skeleton_arr[:, 0:points:2] = Xx
    skeleton_arr[:, 1:points:2] = Xy
    return skeleton_arr


def _pad_data(skeleton_arr: np.ndarray) -> np.ndarray:
    """
    Since decoder is a autoregressive, need to pad the first row as 0s to signify a <start> token.
    """
    padded_data = np.zeros((skeleton_arr.shape[0] + 1, skeleton_arr.shape[1]))
    padded_data[1:] = skeleton_arr
    return padded_data


def _add_position_encoding(skeleton_arr: np.ndarray, total_frame_cnt: int) -> np.ndarray:
    """
    Add position encoding element per frame
    """
    with_encoding = np.zeros((skeleton_arr.shape[0], skeleton_arr.shape[1] + 1))
    with_encoding[:, 0:skeleton_arr.shape[1]] = skeleton_arr
    
    position_encoding = np.arange(0, skeleton_arr.shape[0], 1).astype('float') / total_frame_cnt
    with_encoding[:, -1] = position_encoding
    return with_encoding

    
def create_2d_dataset():
    """
    creates a list of dictionary
    """
    # list of dictionary {"video_id": str, "frame_cnt": int, "skeletons": [[]]}
    with open(os.path.join(DATA_DIR, 'meta', 'video_id_to_word.json'), 'r') as f:
        video_id_dict = json.load(f)
        
    with open(os.path.join(DATA_DIR, 'interim', 'words.pkl'), 'rb') as f:
        words = list(pickle.load(f))
    with open(os.path.join(DATA_DIR, 'interim', 'embeddings.pkl'), 'rb') as f:
        embeddings = pickle.load(f)
        
    openpose_dir = os.path.join(DATA_DIR, 'raw', 'openpose')
    data = []
    for video_id in os.listdir(openpose_dir):
        
        if video_id not in video_id_dict:
            logger.error(f"Can't find the corresponding word to this video {video_id}")
            continue

        logger.info("Looking at {}".format(video_id))
        video_data = {'video_id': video_id,
                      'skeletons': [],
                      'frame_cnt': 0,
                      'word': video_id_dict[video_id]}
        embedding_index = words.index(video_data["word"])
        video_data["embedding"] = embeddings[embedding_index]
        
        video_dir = os.path.join(openpose_dir, video_id)
        frames = sorted(os.listdir(video_dir))
        total_frame_cnt = 0
        for frame_name in frames:
            f = open(os.path.join(video_dir, frame_name))
            try:
                parsed_data = parse_important_skeleton_data(json.load(f))
                if parsed_data is not None:
                    video_data['skeletons'].append(parsed_data)
                    total_frame_cnt += 1
            except Exception as e:
                logger.error("error while grabbing = {}".format(os.path.join(video_dir, frame_name)))
                logger.error(e)
                
        if len(video_data["skeletons"]) > 10:  # arbitrary number check to signify enough frames.
            video_data["skeletons"] = np.array(video_data["skeletons"])
            video_data["skeletons"] = _normalize_data(video_data["skeletons"])
            video_data["skeletons"] = _pad_data(video_data["skeletons"])
            video_data["skeletons"] = _add_position_encoding(video_data["skeletons"], total_frame_cnt)
            video_data["frame_cnt"] = total_frame_cnt
            data.append(video_data)
    with open(os.path.join(DATA_DIR, 'interim', '2dskeleton.pkl'), 'wb') as output:
        pickle.dump(data, output)
    
    
def create_video_id_mapping():
    with open(os.path.join(DATA_DIR, 'meta', 'WLASL_v0.3.json'), 'r') as f:
        meta = json.load(f)

    video_dict = {}
    for word in meta:
        gloss = word["gloss"]
        video_ids = {i["video_id"]: gloss for i in word["instances"]}
        video_dict = {**video_dict, **video_ids}
    
    with open(os.path.join(DATA_DIR, 'meta', 'video_id_to_word.json'), 'w') as output:
        json.dump(video_dict, output)
        
        
def create_word_embedding_map():
    with open(os.path.join(DATA_DIR, 'interim', 'words.pkl'), 'rb') as f:
        words = list(pickle.load(f))
    with open(os.path.join(DATA_DIR, 'interim', 'embeddings.pkl'), 'rb') as f:
        embeddings = pickle.load(f)
        
    mapper = {}
    for i, w in enumerate(words):
        mapper[w] = embeddings[i]
    
    with open(os.path.join(DATA_DIR, 'meta', 'word_to_embedding.pkl'), 'wb') as output:
        pickle.dump(mapper, output)
        
    

        
        
        
        





