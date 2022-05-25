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
  output = {}
  copied = False
  if pred_score:
    for i, score in enumerate(pred_score):
      if score > threshold and pred_class[i] == "boat": #only boats for now
        copied = True
        newBox = {"bbox": [pred_boxes[i][0][0], pred_boxes[i][0][1], pred_boxes[i][1][0], pred_boxes[i][1][1]], "scores": float(score), "labels": pred_class[i]}
        output = newBox
        shutil.copyfile(file, boatDir+fileName+'.jpg')
  if not copied:
    shutil.copyfile(file, noBoatDir+fileName+'.jpg')
    
  
        
  
  with open(jsonDir + fileName + '.json', 'w') as fp:
    json.dump(output, fp)  
  return fileName, output
    
    
  
# =============================================================================
#   with open(jsonpath, 'w') as fp:
#     json.dump(box_dictionary, fp)  
# =============================================================================

def main():
  jsonDirectory = 'json/'
  withBoatDirectory = 'boat/'
  noBoatDirectory = 'noBoat/'
  folder = 'C:/Users/boesan/Downloads/20220415/'

  for root, dirs, files in os.walk(folder):#get list of *.mp4 files
    for file in files:
        if file.endswith('.mp4'):#loop through files
          print(file)
          newFolder = folder + os.path.splitext(file)[0] + "/"
          if not os.path.isdir(newFolder): #make new folder
            os.makedirs(newFolder)
            parseVideo(file, folder, newFolder) #parse video in pictures
  
  #print(glob.glob(folder + '/*'))
  model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
  model.eval()
  directories= [d for d in os.listdir(folder) if os.path.isdir(folder + d)]
  
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
    fileList = [f for f in os.listdir(pathToCurrentDirectory) if os.path.isfile(pathToCurrentDirectory + f)] #os.listdir(pathToCurrentDirectory) if os.path.isdir(folder + d)
    fileList.sort()
    for file in fileList:
      fileName, output = classifyFile(pathToCurrentDirectory+file, model, j, b, nb)
      directoryOutput[fileName + ".jpg"] = output
    with open(j+'directoryOutput.json', 'w') as fp:
      json.dump(directoryOutput, fp)  
      
     
        
# =============================================================================
#   for _, directory, _ in os.walk(folder): #get list of folders
#     for _,_,files in os.walk(directory):
# =============================================================================
    
  
  
  
# =============================================================================
#     for directory in dirs:#loop through folders
#       print(dirs)
#       for _,_,files in os.walk(directory):
#         for file in files:
#           classifyFile(file, model)
# =============================================================================
      
    #loop through files
      #classify images
      #save classification in json
      #save "water" files in one folder
      #save "boat" files in another
# =============================================================================
#   for i in range(7200):
#     print(i)
#     image = cv2.imread("./frames/frame%d.jpg" %(i))
#     
#     #img = cv2.imread("digitbox.jpg", 0)
#     img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     _, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
#     contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     
#     mask = np.ones_like(img) * 255
#     boxes = []
#     
#     for contour in contours:
#         if cv2.contourArea(contour) > 100:
#             hull = cv2.convexHull(contour)
#             cv2.drawContours(mask, [hull], -1, 0, -1)
#             x,y,w,h = cv2.boundingRect(contour)
#             boxes.append((x,y,w,h))
#     
#     boxes = sorted(boxes, key=lambda box: box[0])
#     
#     mask = cv2.dilate(mask, np.ones((5,5),np.uint8))
#     
#     img[mask != 0] = 255
#     
#     result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     showImage = True
#     
#     for n,box in enumerate(boxes):
#         x,y,w,h = box
#         diag = math.sqrt(w**2+h**2)
# # =============================================================================
# #         if diag > 170 and h > 50 and w < 1150:
# #           showImage = True
# #           print(diag)
# #           print(h)
# #           print(w)
# # =============================================================================
#         cv2.rectangle(result,(x,y),(x+w,y+h),(255,0,0),2)
#         cv2.putText(result, str(n),(x+5,y+17), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255,0,0),2,cv2.LINE_AA)
#     
#     if showImage:
#       cv2.imshow("resut", result)
#       cv2.waitKey(0)
#     
#     #cv2.imwrite('digitbox_result.png', result)
#     
#   cv2.destroyAllWindows()
# =============================================================================
    
if __name__=="__main__":
  main()
