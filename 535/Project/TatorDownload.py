# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 10:35:48 2022

@author: boesan
"""
#pip install python-certifi-win32
#pip install tator
#pip install pandas opencv-python

#I can download the video files with the code below and I can download the segement json file but I can't figure out its format.

import tator
import datetime
import glob
import os

def main():  
  #D:\Temp\Git\MachineLearning\535\Project\tatorDownloads
  outFolder = './tatorDownloads/'
  #get newest file in folder
  list_of_files = glob.glob(outFolder + '*') # * means all if need specific format then *.csv
  latest_file = max(list_of_files, key=os.path.getctime)
  latestFileDateTime = datetime.datetime.fromtimestamp(os.path.getmtime(latest_file))
  apiKey = '3fa1c150ecd7bdec71e12663158f115f1b0c547d'
  api = tator.get_api(host='https://cloud.tator.io', token=apiKey)
  #smartPassProjectId = 25
  newportProjectId = 90
  #projects = api.get_project_list()
  medias = api.get_media_list(newportProjectId)
  for media in medias:
    createdDate = datetime.datetime.strptime(media.created_datetime, '%Y-%m-%dT%H:%M:%S.%fZ') 
    #archival = media.media_files.archival
    if createdDate > latestFileDateTime and createdDate > datetime.datetime(2022, 6, 22) and media.codec == 'h264':
      outPath = outFolder + media.name
      outPath = outPath.replace(":","")
      for progress in tator.util.download_media(api, media, outPath):
        print(f"Download progress: {progress}%")
      



if __name__=="__main__":
  main()
