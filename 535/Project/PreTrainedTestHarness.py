
import json
import torch
import cv2
import time
import torchvision
import numpy as np
import statistics
import math

from PIL import Image
from torchvision import transforms as T

vggJsonFile = './via_project_07Jun2022_19h40m26s.json'
pretrainedTestFolder = './PreTrainedTest/'

debugging = False

def main():
  
  
  #list of models: https://pytorch.org/hub/research-models/compact
  #These work
# =============================================================================
  #model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
  model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True) 
  #model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True) 
  #model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True) #doesn't work at all the boxes are waaaay off
# =============================================================================


  #Run but either can't figure out how to compare or don't classify   
# =============================================================================
#   feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-50-panoptic')
#   model = DetrForSegmentation.from_pretrained('facebook/detr-resnet-50-panoptic')
#   model = torch.hub.load('datvuthanh/hybridnets', 'hybridnets', pretrained=True)
#   model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=True)
#   model = Segformer(
#       dims = (32, 64, 160, 256),      # dimensions of each stage
#       heads = (1, 2, 5, 8),           # heads of each stage
#       ff_expansion = (8, 8, 4, 4),    # feedforward expansion factor of each stage
#       reduction_ratio = (8, 4, 2, 1), # reduction ratio of each stage for efficient attention
#       num_layers = 2,                 # num layers of each stage
#       decoder_dim = 256,              # decoder dimension
#       num_classes = 4                 # number of segmentation classes
#   )
#   #https://pytorch.org/vision/stable/generated/torchvision.models.segmentation.fcn_resnet101.html#torchvision.models.segmentation.fcn_resnet101
#   #https://colab.research.google.com/github/pytorch/pytorch.github.io/blob/master/assets/hub/pytorch_vision_fcn_resnet101.ipynb
#   #model = torchvision.models.segmentation.fcn_resnet101(pretrained=True)
# =============================================================================
  
  model.eval()
  transform = T.Compose([T.ToTensor()]) # Defing PyTorch Transform
  
  processedImages = {"records":[]}
  
  accuracies = []
  with open(vggJsonFile) as data_file:    
     data = json.load(data_file)
  fileKeys = {k: data['file'][k] for k in list(data['file'])[1:1000]} #1087 two boats
  for i, fileKey in enumerate(data['file']):  #loop through meta data, all keys
  #for i, fileKey in enumerate(fileKeys):  #testKeys
    #fid = data['metadata'][key]['vid'] #get record id 
    fileName = data['file'][fileKey]['fname']#get file name
    if fileName == '0037_2022-04-15T07_00_00.349178-07_00.jpg': #miss classified
      continue
    if i > 3:
      continue
    if(debugging):
      print(i, fileName)
    boxesInThisImage = []
    for metaKey in data['metadata']:
      #print(data['metadata'][metaKey])
      if data['metadata'][metaKey]['vid'] == fileKey:
        actualRect = data['metadata'][metaKey]['xy'][1:] #get bounding box, first coord is class num always boat
        if(len(actualRect) != 0):
          boxesInThisImage.append([actualRect[0], actualRect[1], actualRect[0]+actualRect[2], actualRect[1]+actualRect[3]])
    if len(boxesInThisImage) > 0:
      actualImage = Image.open(pretrainedTestFolder + fileName) # Load the image
      predictedBoxes = []
      #YOLO
# =============================================================================
#       processedImages['records'].append(actualImage)
#       pred = model([actualImage])
#       for temp in pred.pandas().xyxy:
#         for result in temp.to_numpy():
#           if int(result[5]) == 8:
#            pred_box = [float(result[0]), float(result[1]), float(result[2]), float(result[3])]
#            predictedBoxes.append(pred_box)
# =============================================================================
           
           #non-YOLO    
       
      img = transform(actualImage) # doesn't work for YOLO, comment out
      processedImages['records'].append(img)
      pred = model([img])  # includes NMS
      pred_class = list(pred[0]['labels'].numpy()) # Get the Prediction Score
      pred_boxes = [[(float(i[0]), float(i[1])), (float(i[2]), float(i[3]))] for i in list(pred[0]['boxes'].detach().numpy())] # Bounding boxes
      pred_score = list(pred[0]['scores'].detach().numpy())
      for i, score in enumerate(pred_score):
        if(debugging):
          print(score, pred_class[i])
        if score > .8 and pred_class[i] == 9:
          predictedBoxes.append([pred_boxes[i][0][0], pred_boxes[i][0][1], pred_boxes[i][1][0], pred_boxes[i][1][1]])
          #accuracy = intersectionOverUnion(np.array(actualRect), np.array(pred_box))
# =============================================================================
#           if(accuracy == 0):
#             print(i, fileName)
#           accuracies.append(accuracy)
# =============================================================================
                
      closestBox = [0,0,0,0]
      closestDist = 2000
      for pred_box in predictedBoxes:
        for act_box in boxesInThisImage:
          act_cent = (act_box[0]+(act_box[2]/2), act_box[1]+(act_box[3]/2))
          pred_cent = (pred_box[0]+(pred_box[2]/2), pred_box[1]+(pred_box[3]/2))
          distance = math.sqrt((pred_cent[0] - act_cent[0])**2 + (pred_cent[1] - act_cent[1])**2) 
          if distance < closestDist:
            closestBox = act_box
        accuracies.append(intersectionOverUnion(np.array(closestBox), np.array(pred_box)))
        if(debugging):
          showImage(pretrainedTestFolder + fileName, closestBox, pred_box)
       

        
    #do intersection over union
    
    #pred.show()
    #print(pred_class, pred_boxes, pred_score)
    
    #classify file
    #calc dist
# =============================================================================
#   img = img.convert("RGB")
#   preprocess = T.Compose([
#       T.ToTensor(),
#       T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#   ])
# =============================================================================
  
  #input_tensor = preprocess(img)
  #input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

  #img = img.unsqueeze(0) #needed for hybridnets
  t0 = time.time()
  results = model(processedImages['records'])  # includes NMS
  t1 = time.time()
  print("time:" + str(t1-t0))
  print(accuracies)
  print("accuracies: " + str(statistics.mean(accuracies)))
  
def showImage(filePath, actualRect, predRect):
  rawImg = cv2.imread(filePath)
  cv2.rectangle(rawImg, (int(actualRect[0]), int(actualRect[1])), (int(actualRect[2]), int(actualRect[3])), (0, 255, 0)) #green
  
  cv2.rectangle(rawImg, (int(predRect[0]), int(predRect[1])), (int(predRect[2]), int(predRect[3])), (0, 0, 255)) #red
  cv2.imshow('image', rawImg)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  
#https://stackoverflow.com/questions/28723670/intersection-over-union-between-two-detections
def intersectionOverUnion(box1, box2):
  """
  calculate intersection over union cover percent
  :param box1: box1 with shape (N,4) or (N,2,2) or (2,2) or (4,). first shape is preferred
  :param box2: box2 with shape (N,4) or (N,2,2) or (2,2) or (4,). first shape is preferred
  :return: IoU ratio if intersect, else 0
  """
  # first unify all boxes to shape (N,4)
  if box1.shape[-1] == 2 or len(box1.shape) == 1:
    box1 = box1.reshape(1, 4) if len(box1.shape) <= 2 else box1.reshape(box1.shape[0], 4)
  if box2.shape[-1] == 2 or len(box2.shape) == 1:
    box2 = box2.reshape(1, 4) if len(box2.shape) <= 2 else box2.reshape(box2.shape[0], 4)
  point_num = max(box1.shape[0], box2.shape[0])
  b1p1, b1p2, b2p1, b2p2 = box1[:, :2], box1[:, 2:], box2[:, :2], box2[:, 2:]
  # mask that eliminates non-intersecting matrices
  base_mat = np.ones(shape=(point_num,))
  base_mat *= np.all(np.greater(b1p2 - b2p1, 0), axis=1)
  base_mat *= np.all(np.greater(b2p2 - b1p1, 0), axis=1)  
  intersect_area = np.prod(np.minimum(b2p2, b1p2) - np.maximum(b1p1, b2p1), axis=1)# I area
  union_area = np.prod(b1p2 - b1p1, axis=1) + np.prod(b2p2 - b2p1, axis=1) - intersect_area# U area
  intersect_ratio = intersect_area / union_area# IoU
  return float(base_mat * intersect_ratio)
    
if __name__=="__main__":
  main()
