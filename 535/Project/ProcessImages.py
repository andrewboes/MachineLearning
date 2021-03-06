import cv2 #pip install opencv-python
#import matplotlib.image as img
from matplotlib import pyplot as plt
import numpy as np
import math
from PIL import Image
import torchvision
from datetime import datetime
from torchvision import transforms as T
import os
import torch
import json
import shutil
import collections


COCO_INSTANCE_CATEGORY_NAMES = [
  '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
  'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
  'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
  'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
  'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
  'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
  'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
  'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
  'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
  'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
  'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
  'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

threshold = .8

def parseVideo(videoFile, videoFolder, folder):
  print(videoFolder + videoFile)
  vidcap = cv2.VideoCapture(videoFolder + videoFile)
  success,image = vidcap.read()
  count = 0
  while success:
    #"My name is {fname}, I'm {age}".format(fname = "John", age = 36)
    newFile = "{folder}{image}_{name}.jpg".format(folder = folder, image = "%04d" % count, name= os.path.splitext(videoFile)[0])
    cv2.imwrite(newFile, image)     # save frame as JPEG file      
    success,image = vidcap.read()
    if count % 100 == 0:
      current_time = datetime.now().strftime("%H:%M:%S")
      print('Read a new frame: ', count, current_time)
    count += 1

def get_prediction(img_path, model):
  #print(img_path)
  img = Image.open(img_path) # Load the image
  transform = T.Compose([T.ToTensor()]) # Defing PyTorch Transform
  img = transform(img) # Apply the transform to the image
  pred = model([img]) # Pass the image to the model
  pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())] # Get the Prediction Score
  pred_boxes = [[(float(i[0]), float(i[1])), (float(i[2]), float(i[3]))] for i in list(pred[0]['boxes'].detach().numpy())] # Bounding boxes
  pred_score = list(pred[0]['scores'].detach().numpy())
  return pred_boxes, pred_class, pred_score

def classifyFile(file, model, jsonDir, boatDir, noBoatDir):
  print(file)
  pred_boxes, pred_class, pred_score = get_prediction(file, model)
  fileName = os.path.splitext(os.path.basename(file))[0]
  key = os.path.basename(file)
  output = {}
  output[key] = []
  hasBoat = False
  if pred_score:
    for i, score in enumerate(pred_score):
      if score > threshold and pred_class[i] == "boat": #only boats for now
        hasBoat = True
        newBox = {"bbox": [pred_boxes[i][0][0], pred_boxes[i][0][1], pred_boxes[i][1][0], pred_boxes[i][1][1]], "scores": float(score), "labels": pred_class[i]}
        if len(output[key]) == 0:
          output[key] = [newBox]  
        else:
          output[key].append(newBox)
        
  if hasBoat:
    shutil.copyfile(file, boatDir+fileName+'.jpg')
  else:
    shutil.copyfile(file, noBoatDir+fileName+'.jpg')
  with open(jsonDir + fileName + '.json', 'w') as fp:
    json.dump(output, fp)  
  return fileName, output
    
def main():
  jsonDirectory = 'json/'
  withBoatDirectory = 'boat/'
  noBoatDirectory = 'noBoat/'
  littleBoats = 'littleBoats/'
  folder = 'C:/Users/boesan/Downloads/20220415/'

  #Parse .mp4 files
  # =============================================================================
  for root, dirs, files in os.walk(folder):#get list of *.mp4 files
    for file in files:
        if file.endswith('.mp4'):#loop through files
          print(file)
          newFolder = folder + os.path.splitext(file)[0] + "/"
          if not os.path.isdir(newFolder): #make new folder
            os.makedirs(newFolder)
            parseVideo(file, folder, newFolder) #parse video in pictures
  # =============================================================================
  
  directories= [d for d in os.listdir(folder) if os.path.isdir(folder + d)]
  
  #Classify .jpgs
  # =============================================================================
  model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
  model.eval() 
  
  for directory in directories:
    directoryOutput = {}
    pathToCurrentDirectory = folder+directory + '/'
    j = pathToCurrentDirectory +jsonDirectory
    b = pathToCurrentDirectory + withBoatDirectory
    nb = pathToCurrentDirectory + noBoatDirectory
    if not os.path.isdir(j): #make new folder
      os.makedirs(j)
    if not os.path.isdir(b): #make new folder
      os.makedirs(b)
    if not os.path.isdir(nb): #make new folder
      os.makedirs(nb)
    fileList = [f for f in os.listdir(pathToCurrentDirectory) if os.path.isfile(pathToCurrentDirectory + f)]
    fileList.sort()
    for i, file in enumerate(fileList):
      fileName, output = classifyFile(pathToCurrentDirectory+file, model, j, b, nb)
      directoryOutput[fileName + ".jpg"] = output[fileName + ".jpg"]
    with open(j+'directoryOutput.json', 'w') as fp:
      json.dump(directoryOutput, fp)  
    # =============================================================================
    
  #get 'small' boat pics
  # =============================================================================
  for directory in directories:
  #load 'master' json file
    fullDirectoryJson = folder + directory + '/' +jsonDirectory +'directoryOutput.json'
    with open(fullDirectoryJson) as data_file:    
       data = json.load(data_file)
    odata = collections.OrderedDict(sorted(data.items()))
    littleBoatsFolder = folder + directory + '/' + littleBoats   
    if not os.path.isdir(littleBoatsFolder): #make new folder
      os.makedirs(littleBoatsFolder)
    for i, key in enumerate(odata.keys()):   
      file = folder + directory + '/' + key
      for info in data[key]:
        labels = info.get('labels')
        if labels == "boat": # label 1 is a person in MS COCO Dataset, 9 is boat
          bbox = info.get('bbox')
          print(bbox)
          img = cv2.imread(file)
          print(bbox[2])
          crop_img = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
          newname = littleBoatsFolder + str(i) +"_" + os.path.basename(file)
          cv2.imwrite(newname,crop_img)
          
    # =============================================================================
          


      

    
    # =============================================================================
    
if __name__=="__main__":
  main()
