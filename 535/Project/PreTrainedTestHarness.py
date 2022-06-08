
import json
import torch


vggJsonFile = './via_project_07Jun2022_19h40m26s.json'
pretrainedTestFolder = './PreTrainedTest'

def main():
  
  model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
  with open(vggJsonFile) as data_file:    
     data = json.load(data_file)
  for key in data['metadata']:  #loop through meta data
    fid = data['metadata'][key]['vid'] #get record id 
    fileName = data['file'][fid]['fname']#get file name
    classRect = data['metadata'][key]['xy'][1:] #get bounding box, first coord is class num always boat
    #classify file
    #calc dist
    
if __name__=="__main__":
  main()
