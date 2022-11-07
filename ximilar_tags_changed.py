import cv2
import base64
import json
import os
import warnings
import numpy as np
import random
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from data.base_dataset import Normalize_image
from utils.saving_utils import load_checkpoint_mgpu
from networks import U2NET
import asyncio
import nats                 
from PIL import Image
from io import BytesIO
from ximilar.client import FashionTaggingClient
import time
import pandas as pd
from pillow_heif import register_heif_opener
from google.cloud import storage
import urllib.request
from urllib.parse import quote

def create_wardrobe(df):
  wardrobe = dict()
  wardrobe['Top'] = dict()
  wardrobe['Bottom'] = dict()
  for i in df.index:
    #print(str(df['features_orgimg'][i]))
    ##Women
    tags_noseg = eval(df['features_orgimg'][i])
    for a in tags_noseg['apparels']:
        if len(a['category']) > 0:
            if a['category'][0] == 'Top':
                if len(a['subcategory']) > 0:
                    if a['subcategory'][0] not in wardrobe['Top']:
                        wardrobe['Top'][a['subcategory'][0]] = dict()
                        if len(a['color']) > 0:
                            wardrobe['Top'][a['subcategory'][0]][a['color'][0]] = list()
                            wardrobe['Top'][a['subcategory'][0]][a['color'][0]].append(tags_noseg['url'])
                    else:
                        if len(a['color']) > 0:
                            if a['color'][0] not in wardrobe['Top'][a['subcategory'][0]]:
                                wardrobe['Top'][a['subcategory'][0]][a['color'][0]] = list()
                                wardrobe['Top'][a['subcategory'][0]][a['color'][0]].append(tags_noseg['url'])
                            else:
                                wardrobe['Top'][a['subcategory'][0]][a['color'][0]].append(tags_noseg['url'])
            else:
                if len(a['subcategory']) > 0:
                    if len(a['fit']) > 0 and (a['subcategory'][0] == 'jeans'):
                        if a['subcategory'][0]+":"+a['fit'][0] not in wardrobe['Bottom']:
                            wardrobe['Bottom'][a['subcategory'][0]+":"+a['fit'][0]] = dict()
                            if len(a['color']) > 0:
                                wardrobe['Bottom'][a['subcategory'][0]+":"+a['fit'][0]][a['color'][0]] = list()
                                wardrobe['Bottom'][a['subcategory'][0]+":"+a['fit'][0]][a['color'][0]].append(tags_noseg['url'])
                        else:
                            if len(a['color']) > 0:
                                if a['color'][0] not in wardrobe['Bottom'][a['subcategory'][0]+":"+a['fit'][0]]:
                                    wardrobe['Bottom'][a['subcategory'][0]+":"+a['fit'][0]][a['color'][0]] = list()
                                    wardrobe['Bottom'][a['subcategory'][0]+":"+a['fit'][0]][a['color'][0]].append(tags_noseg['url'])
                            else:
                                wardrobe['Bottom'][a['subcategory'][0]+":"+a['fit'][0]][a['color'][0]].append(tags_noseg['url'])
                    else:
                        if a['subcategory'][0] not in wardrobe['Bottom']:
                            wardrobe['Bottom'][a['subcategory'][0]] = dict()
                            if len(a['color']) > 0:
                                wardrobe['Bottom'][a['subcategory'][0]][a['color'][0]] = list()
                                wardrobe['Bottom'][a['subcategory'][0]][a['color'][0]].append(tags_noseg['url'])
                        else:
                            if len(a['color']) > 0:
                                if a['color'][0] not in wardrobe['Bottom'][a['subcategory'][0]]:
                                    wardrobe['Bottom'][a['subcategory'][0]][a['color'][0]] = list()
                                    wardrobe['Bottom'][a['subcategory'][0]][a['color'][0]].append(tags_noseg['url'])
                            else:
                                wardrobe['Bottom'][a['subcategory'][0]][a['color'][0]].append(tags_noseg['url'])

def get_tags_all_seg(img, device, net, fashion_client, path):

  cascPath = "haarcascade_frontalface_default.xml"
  faceCascade = cv2.CascadeClassifier(cascPath)

  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  faces = faceCascade.detectMultiScale(
  gray,
  scaleFactor=1.1,
  minNeighbors=5,
  flags=cv2.CASCADE_SCALE_IMAGE
  )
  face_count = len(faces)

  get_cloth_segment(path, device = device, net = net)
  print("Image segmentation is done")
  tags = get_tags('image_segment.png', fashion_client=fashion_client)
  #to_bytes = tags.encode()
  return tags
  #else:
    #print("Use Cloth segmentation and yolov5")

def get_tags_all_noseg(img, device, net, fashion_client, path):

  cascPath = "haarcascade_frontalface_default.xml"
  faceCascade = cv2.CascadeClassifier(cascPath)

  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  faces = faceCascade.detectMultiScale(
  gray,
  scaleFactor=1.1,
  minNeighbors=5,
  flags=cv2.CASCADE_SCALE_IMAGE
  )
  face_count = len(faces)

  #if face_count <= 1:
  tags = get_tags(path, fashion_client=fashion_client)
  #to_bytes = tags.encode()
  return tags
  #else:
   # print("Use Cloth segmentation and yolov5")

def get_tags(image_path, fashion_client):
  print("image path in get_tags ", image_path)
  image = cv2.imread(image_path)
  retval, buffer = cv2.imencode('.jpg', image)
  encoded_string = base64.b64encode(buffer).decode('utf-8')
  result = fashion_client.detect_tags_all([{"_base64": encoded_string}])
  list_of_taged_items = result["records"][0]["_objects"]
  all_tags = list()

  for item in list_of_taged_items:
    if 'Category' in item and 'Top Category' in item:
      if item["Top Category"] == 'Clothing' or item['Category'] == "Underwear/Tights":
        try:
          tags = dict() 
          tags = item["_tags"]
          simple_tags = dict()
          for tag, detail in tags.items():
            simple_tags[tag] = list()
            for i in detail:
              simple_tags[tag].append(i["name"])
          all_tags.append(simple_tags)
        except Exception as e:
          continue

  required_tags = dict()
  required_tags['apparels'] = list()
  for item in all_tags:
    d = dict()
    d['category'] = list()
    d['color'] = list()
    d['subcategory'] = list()
    d['fit'] = list()
    if "Category" in item:
      for i in item["Category"]:
        if i == 'Clothing/Upper' or i == 'Clothing/Dresses' or i == 'Clothing/Jackets and Coats':
          d["category"].append("Top")
        else:
           d["category"].append("Bottom")
    if "Color" in item:
      for i in item["Color"]:
        d["color"].append(i)
    if "Subcategory" in item:
      for i in item["Subcategory"]:
        if i != 'cardigans' or i != 'knitted vests' or i != 'vests' or i != 'swim shorts' or i != 'sportswear skirts' or i != 'tutu':
          d["subcategory"].append(i)
    if "Fit" in item:
      for i in item["Fit"]:
        d["fit"].append(i)

    required_tags['apparels'].append(d)
  #required_tags = json.dumps(required_tags)
  required_tags['user_id'] = 'user_men'
  required_tags['skin_tone'] = 3
  required_tags['body_shape'] = 'rectangle'
  required_tags['genere'] = 'western_wear'
  required_tags['gender'] = 'men'
  occasions = ["Keep It Casual", "Just Brunchin\'", "Work", "Date Night", "Party"]
  required_tags['occasions'] = occasions[random.randint(0, len(occasions) - 1)]
  return required_tags 

def get_cloth_segment(image_path, device, net):
  do_palette = True

  def get_palette(num_cls):
    """Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= ((lab >> 0) & 1) << (7 - i)
            palette[j * 3 + 1] |= ((lab >> 1) & 1) << (7 - i)
            palette[j * 3 + 2] |= ((lab >> 2) & 1) << (7 - i)
            i += 1
            lab >>= 3
    return palette
  
  transforms_list = []
  transforms_list += [transforms.ToTensor()]
  transforms_list += [Normalize_image(0.5, 0.5)]
  transform_rgb = transforms.Compose(transforms_list)

  palette = get_palette(4)
  
  with torch.no_grad():
    img = Image.open(image_path).convert("RGB")
    image_tensor = transform_rgb(img)
    image_tensor = torch.unsqueeze(image_tensor, 0).to(device)

    output_tensor = net(image_tensor)
    output_tensor = F.log_softmax(output_tensor[0], dim=1).cpu()
    output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
    output_tensor = torch.squeeze(output_tensor, dim=0)
    output_tensor = torch.squeeze(output_tensor, dim=0)
    output_arr = output_tensor.numpy()

    output_img = Image.fromarray(output_arr.astype("uint8"), mode="L")
    if do_palette:
      output_img.putpalette(palette)
    output_img.save("seg_image.png")

    ##Getting Cloth segment
    img = cv2.imread("seg_image.png")
    ## getting cloth mask to apply
    yellow_pixels = np.where(
        (img[:, :, 0] == 0) & 
        (img[:, :, 1] == 128) & 
        (img[:, :, 2] == 128)
        )

    green_pixels = np.where(
        (img[:, :, 0] == 0) & 
        (img[:, :, 1] == 128) & 
        (img[:, :, 2] == 0)
      )

    red_pixels = np.where(
        (img[:, :, 0] == 0) & 
        (img[:, :, 1] == 0) & 
        (img[:, :, 2] == 128)
    )

    img[yellow_pixels] = [255, 255, 255]
    img[green_pixels] = [255, 255, 255]
    img[red_pixels] = [255, 255, 255]

    binary_mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ## binary mask with 3 channels 
    binary_mask_3 = np.dstack((binary_mask,binary_mask,binary_mask))
    original_image = cv2.imread(image_path)
        
    output_image = np.where(binary_mask_3, original_image, 255)
    
    cv2.imwrite('image_segment.png', output_image)

async def run(device, net, fashion_client):
    nc = await nats.connect(servers=["nats://216.48.179.152:4222"])
    js = nc.jetstream()

    await js.add_stream(name='user10', subjects=['user.id.apparel.feature_men'])
    register_heif_opener()

    MAXHEIGHT = 720
    storage_client = storage.Client()
    blobs = storage_client.list_blobs('test-wardrobe1')

    for blob in blobs:
        ##Escape space
        escape_space = quote(blob.name)
        df = pd.read_csv('wardrobe_data_test.csv')
        urllib.request.urlretrieve('https://storage.googleapis.com/test-wardrobe1/{}'.format(escape_space),"test.png")
        image = Image.open("test.png")
        s = image.size
        ratio = MAXHEIGHT/s[1]
        image = image.resize((int(s[0]*ratio), MAXHEIGHT))
        image = image.save("test.png")
        image = np.array((Image.open("test.png")).convert('RGB'))
        tags_seg = get_tags_all_seg(img = image, device = device, net = net, fashion_client=fashion_client, path = "test.png")
        tags_noseg = get_tags_all_noseg(img = image, device = device, net = net, fashion_client=fashion_client, path = "test.png")
        #image.save(image_byte_arr, format='PNG')
        #image_byte_arr = image_byte_arr.getvalue()
        url =  'https://storage.googleapis.com/test-wardrobe1/{}'.format(escape_space)
        tags_noseg['url'] = url
        df.loc[len(df.index)] = [url, url, tags_seg, tags_noseg]
        df.to_csv('wardrobe_data_test.csv', index=False)
        #print(tags_noseg)
        tags_noseg = json.dumps(tags_noseg)
        to_bytes = tags_noseg.encode()
        ack = await js.publish("user.id.apparel.feature_men", to_bytes)
        print(ack)

    """async def cb(msg):
      start = time.time()
      image = np.array((Image.open(BytesIO(msg.data)).convert('RGB')))
      RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      print("Got image from publisher")
      df = pd.read_csv('wardrobe_data_test.csv')
      image_list = os.listdir('resized-images')
      while True:
        a = random.randint(0, 100)
        if str(a)+'.png' not in image_list:
          cv2.imwrite('resized-images/{}.png'.format(a), RGB_img)
          image_name = 'resized-images/{}.png'.format(a)
          break
      ## may img = image not rgb image
      tags_seg = get_tags_all_seg(img = image, device = device, net = net, fashion_client=fashion_client, path = image_name)
      tags_noseg = get_tags_all_noseg(img = image, device = device, net = net, fashion_client=fashion_client, path = image_name)
      df.loc[len(df.index)] = [image_name, image_name, tags_seg, tags_noseg]
      df.to_csv('wardrobe_data_test.csv', index=False)
      tags_noseg = json.dumps(tags_noseg)
      to_bytes = tags_noseg.encode()

      ack = await js.publish("user.id.apparel.feature", to_bytes)
      print(ack)
      await msg.ack()

      end = time.time()
      print("The time of execution of above program is :",(end-start) * 10**3, "ms")"""
        
    #await js.subscribe("user.id.apparel.test2", cb=cb)

if __name__ == '__main__':

    ##Load Model
    device = "cuda:0"
    checkpoint_path = 'cloth_segm_u2net_latest.pth'
    net = U2NET(in_ch=3, out_ch=4)
    net = load_checkpoint_mgpu(net, checkpoint_path)
    net = net.to(device)
    net = net.eval()

    fashion_client = FashionTaggingClient(token="2f7c51901dda0937e4c991938fcbbba6f011f275")

    loop = asyncio.get_event_loop()
    loop.run_until_complete(run(device = device, net = net, fashion_client = fashion_client))
    loop.run_forever()

