#references:
#https://colab.research.google.com/github/mlvlab/COSE474/blob/master/3_Object_Detection_and_MOT_tutorial.ipynb#scrollTo=cnS6Q7M7C2VP
#https://learnopencv.com/faster-r-cnn-object-detection-with-pytorch/
#https://towardsdatascience.com/object-detection-and-tracking-in-pytorch-b3cf1a696a98
#SORT paper: https://arxiv.org/pdf/1602.00763.pdf
#SORT code: https://github.com/abewley/sort

import os 
from PIL import Image
import numpy as np
import torch
import torchvision
from torchvision import transforms as T
import matplotlib.pylab as plt
import cv2
import sys
import skimage
import json
import collections
from pprint import pprint
from sort import *

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" #hack

mot = "./MOT_boat/"   # a custom path. you can change if you want to
MOT_PATH = mot
#motdata = MOT_PATH + 'train/MOT17-09/img1/' #people
motdata = MOT_PATH + 'data/' #boats
jsonpath =   MOT_PATH + 'data.json'
save_path = MOT_PATH + 'save/'
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

def get_prediction(img_path, model):
  #img_path = "./MOT_boat/data/data0009.jpg"
  print(img_path)
  img = Image.open(img_path) # Load the image
  transform = T.Compose([T.ToTensor()]) # Defing PyTorch Transform
  img = transform(img) # Apply the transform to the image
  pred = model([img]) # Pass the image to the model
  pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())] # Get the Prediction Score
  pred_boxes = [[(float(i[0]), float(i[1])), (float(i[2]), float(i[3]))] for i in list(pred[0]['boxes'].detach().numpy())] # Bounding boxes
  pred_score = list(pred[0]['scores'].detach().numpy())
  return pred_boxes, pred_class, pred_score
  

def object_detection_api(img_path, model, threshold=0.5, rect_th=3, text_size=1.5, text_th=3):
  
  boxes, pred_cls, pred_score = get_prediction(img_path, threshold, model) # Get predictions
  if boxes:
    img = cv2.imread(img_path) # read image with cv2
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB
    for i in range(len(boxes)):
     cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th) # Draw Rectangle with the coordinates
     cv2.putText(img,pred_cls[i], boxes[i][0],  cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th) # Write the prediction class
    cv2.imwrite("./MOT_boat/detect.jpg", img)   
 

 
def main():  
  threshold = .8
  list_motdata = os.listdir(motdata)  
  list_motdata.sort()
     
  model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
  model.eval()

  img_ex_path = motdata + list_motdata[0]
  #img_ex_origin = cv2.imread(img_ex_path)
  #img_ex = cv2.cvtColor(img_ex_origin, cv2.COLOR_BGR2RGB)
  #object_detection_api(img_ex_path, model, threshold=0.8)
  box_dictionary = {}
  
  for x in list_motdata[530:570]: #530:570
    img_ex_path = motdata + x
    #img_ex_origin = cv2.imread(img_ex_path)
    #img_ex = cv2.cvtColor(img_ex_origin, cv2.COLOR_BGR2RGB)
    pred_boxes, pred_class, pred_score = get_prediction(img_ex_path, model)
    box_dictionary[x] = [] 
    if pred_score:
      for i, score in enumerate(pred_score):
        if score > threshold and pred_class[i] == "boat": #only boats for now
          newBox = {"bbox": [pred_boxes[i][0][0], pred_boxes[i][0][1], pred_boxes[i][1][0], pred_boxes[i][1][1]], "scores": float(score), "labels": pred_class[i]}
          if len(box_dictionary[x]) == 0:
            box_dictionary[x] = [newBox]  
          else:
            box_dictionary[x].append(newBox)
    
  
  with open(jsonpath, 'w') as fp:
    json.dump(box_dictionary, fp)  
  
  with open(jsonpath) as data_file:    
     data = json.load(data_file)
  odata = collections.OrderedDict(sorted(data.items()))
  #pprint(odata)
  img_path = motdata    # img root path

# Making new directory for saving results
  
  mot_tracker = Sort() 
  for key in odata.keys():   
    arrlist = []
    det_img = cv2.imread(os.path.join(img_path, key))
    #overlay = det_img.copy()
    det_result = data[key] 
    
    for info in det_result:
      labels =  info.get('labels')
      if labels == "boat": # label 1 is a person in MS COCO Dataset, 9 is boat
        bbox = info.get('bbox')
        labels =  info.get('labels')
        scores =  info.get('scores')
        templist = bbox + [scores]
        arrlist.append(templist)
      
    track_bbs_ids = mot_tracker.update(np.array(arrlist))
    
    mot_imgid = key.replace('.jpg','')
    newname = save_path + mot_imgid + '_mot.jpg'
    print(mot_imgid)
    
    for j in range(track_bbs_ids.shape[0]):  
        ele = track_bbs_ids[j, :]
        x = int(ele[0])
        y = int(ele[1])
        x2 = int(ele[2])
        y2 = int(ele[3])
        track_label = str(int(ele[4])) 
        cv2.rectangle(det_img, (x, y), (x2, y2), (0, 255, 255), 4)
        cv2.putText(det_img, '#'+track_label, (x+5, y-10), 0,0.6,(0,255,255),thickness=2)
    
    cv2.imwrite(newname,det_img)

  frameslist = os.listdir(save_path)
  import imageio
  images = []
  for filename in frameslist:
    print(filename)
    images.append(imageio.imread(save_path + filename))
  imageio.mimsave(MOT_PATH + "movie.gif", images, fps=3)
  

if __name__=="__main__":
  main()

