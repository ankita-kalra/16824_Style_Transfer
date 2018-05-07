import numpy as np
import os
import random

#reading random number of images from each class
def getRandomFilenames(path,class_number,num_files):
  """
  Returns a random filename, chosen among the files of the given path.
  """
  for i in range(1,num_files):
      files = os.listdir(path)
      indices_random = random.sample(range(0, len(files)),2)
      selectedfiles = [files[indices_random[0]],files[indices_random[1]]]

      with open(dataset_filepath, 'a') as of:
          line_out = '{} {} {}\r\n'.format(selectedfiles[0], selectedfiles[1],class_number)
          of.write(line_out)

class_names=['amusement_park','beach','highway','house','playground']
num_files = 20
phase = 'train//'
dataset_filepath = "./SunData/train.txt"
path="./SunData/"

for i in range(0,5):
    path = path + phase+ class_names[i]
    getRandomFilenames(path,i,num_files)
    path = "./SunData/"
