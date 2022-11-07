import pandas as pd

import json
import os
import warnings
import numpy as np
import random
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import asyncio
import nats                 


df = pd.read_csv('wardrobe_data_test.csv')
wardrobe = dict()
wardrobe['Top'] = dict()
wardrobe['Bottom'] = dict()
count = 0
for i in range(210, 480):
    count = count+1
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
                    is_a_fit_type = False
                    subcat = ''
                    for item in a['subcategory']:
                        if item == 'jeans' or item == 'formal trousers':
                            is_a_fit_type = True
                            subcat = item
                            break
                    if is_a_fit_type == True:
                        if len(a['fit']) > 0:
                            if subcat+":"+a['fit'][0] not in wardrobe['Bottom']:
                                wardrobe['Bottom'][subcat+":"+a['fit'][0]] = dict()
                                if len(a['color']) > 0:
                                    wardrobe['Bottom'][subcat+":"+a['fit'][0]][a['color'][0]] = list()
                                    wardrobe['Bottom'][subcat+":"+a['fit'][0]][a['color'][0]].append(tags_noseg['url'])
                            else:
                                if len(a['color']) > 0:
                                    if a['color'][0] not in wardrobe['Bottom'][subcat+":"+a['fit'][0]]:
                                        wardrobe['Bottom'][subcat+":"+a['fit'][0]][a['color'][0]] = list()
                                        wardrobe['Bottom'][subcat+":"+a['fit'][0]][a['color'][0]].append(tags_noseg['url'])
                                else:
                                    wardrobe['Bottom'][subcat+":"+a['fit'][0]][a['color'][0]].append(tags_noseg['url'])
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

print(wardrobe)


