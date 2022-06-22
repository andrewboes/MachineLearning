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

def main():

  apiKey = ''
  api = tator.get_api(host='https://cloud.tator.io', token=apiKey)
  outFolder = './tatorDownloads/'
  mediaId = 1123613
  projectId = 25
  segmentpath = '14/25/1123613/fb8a73d4-f824-11ea-8c08-06b7c5058ed6_segments.json'
  project = api.get_project(projectId)
  #print(project)
  outpath = outFolder + 'file2.json'
# =============================================================================
#   for progress in _download_file(api, project, segmentpath, outpath):
#     print(f"Download progress: {progress}%")
# =============================================================================
  
  media = api.get_media(mediaId)
  print(media)
# =============================================================================
#   out_path = outFolder + 'test.log'
#   for progress in tator.util.download_attachment(api, media, out_path):
#       print(f"Attachment download progress: {progress}%")
# =============================================================================
  for progress in tator.util.download_media(api, media, outpath):
    print(f"download progress: {progress}%")
# =============================================================================
# projects = api.get_project_list()
# for project in projects:
#   print(project.id)
#   medias = api.get_media_list(project.id)
#   print(len(medias))
# =============================================================================
# =============================================================================
#     for progress in tator.util.download_media(api, media, out_path):
#         print(f"Download progress: {progress}%")
# =============================================================================


import math
import logging
from urllib.parse import urljoin

import requests

logger = logging.getLogger(__name__)

def _download_file(api, project, url, out_path):
    CHUNK_SIZE = 10 * 1024 * 1024
    MAX_RETRIES = 10
    # If this is a normal url, get headers.
    if url.startswith('/'):
        config = api.api_client.configuration
        host = config.host
        token = config.api_key['Authorization']
        prefix = config.api_key_prefix['Authorization']
        url = urljoin(host, url)
        # Supply token here for eventual media authorization
        headers = {
            'Authorization': f'{prefix} {token}',
            'Content-Type': f'application/json',
            'Accept-Encoding': 'gzip',
        }
    elif url.startswith('http'):
        headers = {}
    # If this is a S3 object key, get a download url.
    else:
        url = api.get_download_info(project, {'keys': [url]})[0].url
        headers = {}
    for attempt in range(MAX_RETRIES):
        try:
            with requests.get(url, stream=True, headers=headers) as r:
                r.raise_for_status()
                total_size = r.headers['Content-Length']
                total_chunks = math.ceil(int(total_size) / CHUNK_SIZE)
                chunk_count = 0
                last_progress = 0
                yield last_progress
                with open(out_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            chunk_count += 1
                            f.write(chunk)
                            this_progress = round((chunk_count / total_chunks) *100,1)
                            if this_progress != last_progress:
                                yield this_progress
                                last_progress = this_progress
                yield 100
            break
        except Exception as ex:
            logger.error(f"Failed to download {url} on attempt {attempt} {ex}...")
            pass


if __name__=="__main__":
  main()
