import asyncio
import nats                 
import os
import random 
import time
from PIL import Image
from io import BytesIO
from pymongo import MongoClient
import numpy as np
import cv2
import uuid
from pillow_heif import register_heif_opener
from google.cloud import storage
import urllib.request

async def main():

    ##Connect to server
    nc = await nats.connect(servers=["nats://216.48.179.152:4222"])
    js = nc.jetstream()
    #await js.delete_stream(name="user")

    await js.add_stream(name='user5', subjects=['user.id.apparel.test3'])
    register_heif_opener()

    MAXHEIGHT = 720
    storage_client = storage.Client()
    blobs = storage_client.list_blobs('test-wardrobe')

    for blob in blobs:
        print(blob.name)
        urllib.request.urlretrieve('https://storage.googleapis.com/test-wardrobe/{}'.format(blob.name),"test.png")
        image = Image.open("test.png")
        s = image.size
        ratio = MAXHEIGHT/s[1]
        image = image.resize((int(s[0]*ratio), MAXHEIGHT))
        image_byte_arr = BytesIO()
        image.save(image_byte_arr, format='PNG')
        image_byte_arr = image_byte_arr.getvalue()
        url =  'https://storage.googleapis.com/test-wardrobe/{}'.format(blob.name)
        url = bytes(url, 'utf-8')
        to_bytes = dict()
        to_bytes
        ack = await js.publish("user.id.apparel.test3", image_byte_arr)
        print(ack)

    #await nc.close()

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    try:
        asyncio.ensure_future(main())
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        print("Closing Loop")
        loop.close()
    #asyncio.run(main())