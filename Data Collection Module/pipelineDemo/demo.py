# standard
import h5py

# 3rd party
import numpy

# our own 
import skeletalModel
import pose2D
import pose2Dto3D
import pose3D 
import sys



def save(fname, lst):

  T, dim = lst[0].shape
  f = open(fname, "w")
  for t in range(T):
    for i in range(dim):
      for j in range(len(lst)):
        f.write("%e\t" % lst[j][t, i])
    f.write("\n")
  f.close()


def noNones(l):
  l2 = []
  for i in l:
    if not i is None:
      l2.append(i)
  return l2
  
def main(argv):
  dtype = "float32"
  randomNubersGenerator = numpy.random.RandomState(1234)

  # This demo shows converting a result of 2D pose estimation into a 3D pose.
  
  # Getting our structure of skeletal model.
  # For customizing the structure see a definition of getSkeletalModelStructure.  
  structure = skeletalModel.getSkeletalModelStructure()
  
  # Getting 2D data
  # The sequence is an N-tuple of
  #   (1sf point - x, 1st point - y, 1st point - likelihood, 2nd point - x, ...)
  # a missing point should have x=0, y=0, likelihood=0 
  f = h5py.File("./pipelineDemo/keypoints.h5", "r")
  # todo change the location or folder name later 
  inputSequence_2D = numpy.array(f.get(argv[1]))
  #inputSequence_2D = numpy.array(f.get("sample_2d_openpose"))
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
  save("Temp_Data/"+ argv[1] + "_demo1.txt", [Xx, Xy, Xw])

  # Delete all skeletal models which have a lot of missing parts.
  Xx, Xy, Xw = pose2D.prune(Xx, Xy, Xw, (0, 1, 2, 3, 4, 5, 6, 7), 0.3, dtype)
  save("Temp_Data/"+ argv[1] + "_demo2.txt", [Xx, Xy, Xw])
  
  # Preliminary filtering: weighted linear interpolation of missing points.
  Xx, Xy, Xw = pose2D.interpolation(Xx, Xy, Xw, 0.99, dtype)
  save("Temp_Data/"+ argv[1] + "_demo3.txt", [Xx, Xy, Xw])
  
  # Initial 3D pose estimation
  lines0, rootsx0, rootsy0, rootsz0, anglesx0, anglesy0, anglesz0, Yx0, Yy0, Yz0 = pose2Dto3D.initialization(
    Xx,
    Xy,
    Xw,
    structure,
    0.001, # weight for adding noise
    randomNubersGenerator,
    dtype
  )
  save("Temp_Data/"+ argv[1] + "_demo4.txt", [Yx0, Yy0, Yz0])
    
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
  data = numpy.asarray(noNones([numpy.hstack((Yx,Face_Xx)), numpy.hstack((Yy,Face_Xy)), numpy.hstack((Yz,Face_Xw))]), dtype="float32")
  save("Training_Data_3D/"+ argv[1] + ".txt", data)
  hf = h5py.File("Training_Data_3D_h5/"+argv[1]+".h5", "w")
  hf.create_dataset(argv[1], data=data, dtype="float32")    

  hf.close()

if __name__ == "__main__":
    main(sys.argv)
      
