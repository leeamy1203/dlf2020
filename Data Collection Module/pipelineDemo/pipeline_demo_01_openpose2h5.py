import codecs
import math
import re
import os
import json
import h5py
import subprocess

import numpy
def walkDir(dname, filt=r".*"):
  result = []
  for root, dnames, fnames in os.walk(dname):
      for fname in fnames:
          if re.search(filt, fname):
              foo = root + "/" + fname
              foo = re.sub(r"[/\\]+", "/", foo)
              result.append(foo)
  return result



def selectPoints(points, keepThis):
  points2 = []
  for i in keepThis:
    points2.append(points[3 * i + 0])
    points2.append(points[3 * i + 1])
    points2.append(points[3 * i + 2])
  return points2


def noNones(l):
  l2 = []
  for i in l:
    if not i is None:
      l2.append(i)
  return l2


def loadData(dname):
  fnames = walkDir(dname = dname, filt = r"\.json$")
  fnames.sort()
  frames = []
  for fname in fnames:
    p = re.search(r"([^\\/]+)_(\d+)_keypoints\.json$", fname)
    
    try:
        with open(fname) as json_data:
          data = json.load(json_data)
    except:
        continue
    if len(data["people"]) == 0:
      continue
      
    i = int(p.group(2))
    while len(frames) < i + 1:
      frames.append(None)
    
    theTallest = data["people"][0]
    
    idxsPose = [0, 1, 2, 3, 4, 5, 6, 7]
    idxsHand = range(21)    

    if theTallest is None:
      points = (3 * (len(idxsPose) + 2 * len(idxsHand)) * [0.0], [0.0] * 105 * 3)
    else:
      pointsP = theTallest["pose_keypoints_2d"]
      pointsLH = theTallest["hand_left_keypoints_2d"]
      pointsRH = theTallest["hand_right_keypoints_2d"]
      pointsFace = theTallest["face_keypoints_2d"]
      pointsP = selectPoints(pointsP, idxsPose)
      pointsLH = selectPoints(pointsLH, idxsHand)
      pointsRH = selectPoints(pointsRH, idxsHand)
      points = pointsP + pointsLH + pointsRH + pointsFace

    if not points[0] == 0.0:
      frames[i] = points
    
  return numpy.asarray(noNones(frames), dtype="float32")


if __name__ == "__main__":

  dnameOpenPose = "./pipelineDemo/2DJSON_Keypoints"
  fnameH5 = "./pipelineDemo/keypoints.h5"
  
  
  recs = {}
  for fname in walkDir(dnameOpenPose, filt=r"\.[jJ][sS][oO][nN]$"):
    dname = re.sub(r"(.*)[/\\].*", r"\1", fname)
    key = re.sub(r".*[/\\]", "", dname)
    recs[key] = dname
  
  hf = h5py.File(fnameH5, "w")
  
  for key in recs:
    data = loadData(recs[key])
    hf.create_dataset(key, data=data, dtype="float32")    

  hf.close()
