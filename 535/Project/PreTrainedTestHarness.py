
import json
import torch
import cv2
import time
vggJsonFile = './via_project_07Jun2022_19h40m26s.json'
pretrainedTestFolder = './PreTrainedTest/'

def main():
  
  model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
  
  processedImages = {"records":[]}
  with open(vggJsonFile) as data_file:    
     data = json.load(data_file)
  testKeys = {k: data['metadata'][k] for k in list(data['metadata'])[:100]}
  #for key in data['metadata']:  #loop through meta data, all keys
  for key in testKeys:  #testKeys
    fid = data['metadata'][key]['vid'] #get record id 
    fileName = data['file'][fid]['fname']#get file name
    classRect = data['metadata'][key]['xy'][1:] #get bounding box, first coord is class num always boat
    processedImages['records'].append(cv2.imread(pretrainedTestFolder + fileName)[..., ::-1])
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
  results = model(processedImages['records'], size=640)  # includes NMS
  t1 = time.time()
  print(t1-t0)
    
if __name__=="__main__":
  main()
