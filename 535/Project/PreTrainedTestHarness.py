
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
  #work
  #model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
  #model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True) 
  #https://pytorch.org/vision/stable/generated/torchvision.models.segmentation.fcn_resnet101.html#torchvision.models.segmentation.fcn_resnet101
  #https://colab.research.google.com/github/pytorch/pytorch.github.io/blob/master/assets/hub/pytorch_vision_fcn_resnet101.ipynb
  model = torchvision.models.segmentation.fcn_resnet101(pretrained=True)


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
# =============================================================================
  
  model.eval()
  transform = T.Compose([T.ToTensor()]) # Defing PyTorch Transform
  
  processedImages = {"records":[]}
  with open(vggJsonFile) as data_file:    
     data = json.load(data_file)
  testKeys = {k: data['metadata'][k] for k in list(data['metadata'])[:5]}
  #for key in data['metadata']:  #loop through meta data, all keys
  for key in testKeys:  #testKeys
    fid = data['metadata'][key]['vid'] #get record id 
    fileName = data['file'][fid]['fname']#get file name
    classRect = data['metadata'][key]['xy'][1:] #get bounding box, first coord is class num always boat
    img = Image.open(pretrainedTestFolder + fileName) # Load the image
    #img = cv2.imread(pretrainedTestFolder + fileName)[..., ::-1]
    #img = transform(img) # doesn't work for YOLO, comment out
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
  img = img.convert("RGB")
  preprocess = T.Compose([
      T.ToTensor(),
      T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])
  
  input_tensor = preprocess(img)
  input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

  #img = img.unsqueeze(0) #needed for hybridnets
  t0 = time.time()
  results = model(input_batch)  # includes NMS
  t1 = time.time()
  print(t1-t0)
  
    
if __name__=="__main__":
  main()
