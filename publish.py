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


async def main():

    ##Connect to server
    nc = await nats.connect(servers=["nats://216.48.179.152:4222"])

    await nc.publish("test_whatsapp", b'First')
    await nc.publish("test_whatsapp", b'Second')
    print('published')

    await nc.close()

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