#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###########################################################################
# Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
###########################################################################
import numpy as np
import json
import pandas as pd

def create_json(path, save_json_path):

    bad_files = [
    "test_132.jpg",
    "test_1346.jpg",
    "test_184.jpg",
    "test_1929.jpg",
    "test_2028.jpg",
    "test_22.jpg",
    "test_232.jpg",
    "test_2321.jpg",
    "test_2613.jpg",
    "test_2643.jpg",
    "test_274.jpg",
    "test_2878.jpg",
    "test_521.jpg",
    "test_853.jpg",
    "test_910.jpg",
    "test_923.jpg",
    "train_1239.jpg",
    "train_2376.jpg",
    "train_2903.jpg",
    "train_2986.jpg",
    "train_305.jpg",
    "train_3240.jpg",
    "train_340.jpg",
    "train_3556.jpg",
    "train_3560.jpg",
    "train_38.jpg",
    "train_3832.jpg",
    "train_4222.jpg",
    "train_5007.jpg",
    "train_5137.jpg",
    "train_5143.jpg",
    "train_5762.jpg",
    "train_5822.jpg",
    "train_6052.jpg",
    "train_6090.jpg",
    "train_6138.jpg",
    "train_6409.jpg",
    "train_6722.jpg",
    "train_6788.jpg",
    "train_737.jpg",
    "train_7576.jpg",
    "train_7622.jpg",
    "train_775.jpg",
    "train_7883.jpg",
    "train_789.jpg",
    "train_8020.jpg",
    "train_8146.jpg",
    "train_882.jpg",
    "train_903.jpg",
    "train_924.jpg",
    "val_147.jpg",
    "val_286.jpg",
    "val_296.jpg",
    "val_386.jpg"
    ]

    data = pd.read_csv(path, usecols=[0,1,2,3,4,5,6,7], names=['filename','xmin','ymin','xmax','ymax','class','width','height'], header=None)

    for n in bad_files: 
        data = data.drop(data[data['filename'] == n].index)

    images = []
    categories = []
    annotations = []

    category = {}
    category["supercategory"] = 'none'
    category["id"] = 1
    category["name"] = 'None'
    categories.append(category)

    data['fileid'] = data['filename'].astype('category').cat.codes
    data['categoryid']= pd.Categorical(data['class'],ordered= True).codes
    data['categoryid'] = data['categoryid']+1
    data['annid'] = data.index

    def image(row):
        image = {}
        image["height"] = row.height
        image["width"] = row.width
        image["id"] = row.fileid
        image["file_name"] = row.filename
        return image

    def category(row):
        category = {}
        category["supercategory"] = 'None'
        category["id"] = row.categoryid
        category["name"] = row[2]
        return category

    def annotation(row):
        annotation = {}
        area = 100
        #area = (row.xmax -row.xmin)*(row.ymax - row.ymin)
        annotation["segmentation"] = []
        annotation["iscrowd"] = 0
        annotation["area"] = area
        annotation["image_id"] = row.fileid

        annotation["bbox"] = [row.xmin, row.ymin, row.xmax -row.xmin,row.ymax-row.ymin ]

        annotation["category_id"] = row.categoryid
        annotation["id"] = row.annid
        return annotation

    for row in data.itertuples():
        annotations.append(annotation(row))

    imagedf = data.drop_duplicates(subset=['fileid']).sort_values(by='fileid')
    for row in imagedf.itertuples():
        images.append(image(row))

    catdf = data.drop_duplicates(subset=['categoryid']).sort_values(by='categoryid')
    for row in catdf.itertuples():
        categories.append(category(row))

    data_coco = {}
    data_coco["images"] = images
    data_coco["categories"] = categories
    data_coco["annotations"] = annotations
    json.dump(data_coco, open(save_json_path, "w"), indent=4)
