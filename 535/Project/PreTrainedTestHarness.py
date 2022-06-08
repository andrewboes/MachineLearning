
import json
import torch
import cv2
import time
import torchvision
from PIL import Image

from torchvision import transforms as T
vggJsonFile = './via_project_07Jun2022_19h40m26s.json'
pretrainedTestFolder = './PreTrainedTest/'

def main():
  
  #list of models: https://pytorch.org/hub/research-models/compact
  #model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
  model = torch.hub.load('datvuthanh/hybridnets', 'hybridnets', pretrained=True)
  #model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=True)
  #model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True) 
  model.eval()
  transform = T.Compose([T.ToTensor()]) # Defing PyTorch Transform
  
  processedImages = {"records":[]}
  with open(vggJsonFile) as data_file:    
     data = json.load(data_file)
  testKeys = {k: data['metadata'][k] for k in list(data['metadata'])[:1]}
  #for key in data['metadata']:  #loop through meta data, all keys
  for key in testKeys:  #testKeys
    fid = data['metadata'][key]['vid'] #get record id 
    fileName = data['file'][fid]['fname']#get file name
    classRect = data['metadata'][key]['xy'][1:] #get bounding box, first coord is class num always boat
    img = Image.open(pretrainedTestFolder + fileName) # Load the image
    img = img.resize((640,384))
    myImg = transform(img) # doesn't work for YOLO, comment out
    processedImages['records'].append(img)
    #processedImageIds.append(key)
# =============================================================================
#     pred = model(cv2.imread(pretrainedTestFolder + fileName)[..., ::-1], size=640)  # includes NMS
#     pred_class = list(pred[0]['labels'].numpy()) # Get the Prediction Score
#     pred_boxes = [[(float(i[0]), float(i[1])), (float(i[2]), float(i[3]))] for i in list(pred[0]['boxes'].detach().numpy())] # Bounding boxes
#     pred_score = list(pred[0]['scores'].detach().numpy())
#     pred.show()
#     print(pred_class, pred_boxes, pred_score)
# =============================================================================
    
    #classify file
    #calc dist
  


  t0 = time.time()
  img = torch.randn(1,3,640,384)
  myImg = myImg.unsqueeze(0)
  print(myImg.size())
  features, regression, classification, anchors, segmentation = model(myImg)  # includes NMS
  t1 = time.time()
  print(t1-t0)
    
if __name__=="__main__":
  main()
