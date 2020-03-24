import sys
sys.path.append("/home/aab10867zc/work/aist/pspicker/code")
import config
import utils
import pandas as pd
import numpy as np
from obspy import Trace,Stream
import matplotlib.pyplot as plt
from obspy.core import read
from glob import glob
import shutil
import math
import datetime
import random
import json
import argparse
import os
import keras
import multiprocessing
import re
import itertools
import tensorflow as tf
import keras.backend as K
import matplotlib.patches as patches

import model_multi_confidence_sta_mask_pure as MultiModel
MULTI_MODEL_PATH="/home/aab10867zc/work/aist/pspicker/training_plan/pspicker20191024T145750/4/ckpt-e021-l1.0150.h5"
import model as SingleModel
SINGLE_MODEL_PATH="/home/aab10867zc/work/aist/pspicker/training_plan/pspicker20190828T152936/4/ckpt-e026-l0.2041.h5"

TEST_DICT="/home/aab10867zc/work/aist/pspicker/metadata/pspicker_meta_test_2019-07-29.json"
MODEL_DIR="/home/aab10867zc/work/aist/pspicker/training_plan"
EVAL_DIR="/home/aab10867zc/work/aist/pspicker/evaluation/confidence_mask_sinmul_easy"


#weighted by station
class MultiInferenceConfig(config.Config):

    #multi std 0110
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    NAME="pspicker"
    GPU_COUNT = 1
    WINDOWS_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE=0.5

    DETECTION_NMS_THRESHOLD=0.3
    RPN_ANCHOR_SCALES=[1524, 2436,3648,4860,6072]

    RPN_ANCHOR_RATIOS=[0.5,1,1.5,2]

    DIVISION_SIZE=1028

    WINDOW_STATION_DIM = 10

    RPN_NMS_THRESHOLD = 0.7

    FPN_CLASSIF_FC_LAYERS_SIZE = 1024


    POOL_SIZE = [WINDOW_STATION_DIM,14]
    MASK_POOL_SIZE = [WINDOW_STATION_DIM,28]
    MASK_SHAPE = [WINDOW_STATION_DIM,56]

    BACKBONE_CONV=False
    RPN_CONV=False
    MRCNN_CONV=False

class SingleInferenceConfig(config.Config):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            NAME="pspicker"
            GPU_COUNT = 1
            WINDOWS_PER_GPU = 10
            DETECTION_MIN_CONFIDENCE=0
            RPN_ANCHOR_SCALES=[64, 128, 256, 512, 1024]

            RPN_ANCHOR_RATIOS=[0.5,1,2]

            DIVISION_SIZE=1024

            DETECTION_NMS_THRESHOLD=0.01

            DETECTION_MIN_CONFIDENCE=0.7
            CONV_STATION=False

#neighbour stations
#no substations
#eazy mode (sorted by nearest station order)

class PSpickerDataset(MultiModel.Dataset):
    """Generates the pspicker synthetic dataset. The dataset consists of
    seismic waveform windows of shape (stations,time_width,channels).
    """

    def load_sac(self, sac_info,shape=[10,12000,3],add_sub=True):
        """Load a subset of the pspicker dataset.
        dataset_dir: The root directory of the pspicker dataset.
        subset: What to load (train, val, test)

        return_coco: If True, returns the COCO object.
        """


        # Add classes
        self.add_class("pspicker", 1, "ps")

        for window_id,main_event in enumerate(sac_info["windows"]):

            path = [main_event["traces"][station] for station in main_event["stations"]]

            if len(path)<shape[0]:
                continue

            self.add_window("pspicker",window_id=window_id,main_stations=main_event["stations"],
                            main_name=main_event["name"],shape=shape,path=path)



    def load_streams(self,window_id):
        info = self.window_info[window_id]
        shape=info["shape"]
        streams=[]
        dist = []

        for event in info["path"]:
            paths=list(event.values())
            traces=[]
            for path in paths:
                trace=read(path)[0]
                traces.append(trace)

            stream=Stream(traces=traces)
            stream.detrend("constant")
            stream.filter("highpass", freq=2.0)
            dist.append(stream[0].stats.sac["dist"])

            for i in range(len(stream)):
                stream[i].data-=np.mean(stream[i].data)
                stream[i].data/=np.std(stream[i].data)

            streams.append(stream)

        index=np.argsort(dist)[:10]
        streams = [streams[i] for i in index]



        return streams





    def load_window(self, window_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        streams = self.load_streams(window_id)
        info=self.window_info[window_id]
        shape=info["shape"]
        np.random.seed(window_id)

        window=np.random.normal(0.0,0.1,shape)

        for station,stream in enumerate(streams):

            channel_dict={"U":0,"N":1,"E":2}
            for trace in stream:
                channel=channel_dict[trace.stats.channel]
                npts=min(trace.stats.npts,shape[1])
                window[station,:npts,channel]=trace.data

        if self.shuffle:
            random.seed(window_id)
            random_index=random.sample(range(shape[0]),shape[0])
            window=window[random_index]

        return window


    def window_reference(self, window_id):
        """Return the shapes data of the image."""
        info = self.window_info[window_id]
        if info["source"] == "pspikcer":
            return info["station"]
        else:
            super(self.__class__).window_reference(self, window_id)

    def load_mask(self, window_id):
        """Generate instance masks for shapes of the given image ID.
        """
        streams = self.load_streams(window_id)
        info=self.window_info[window_id]
        shape=info["shape"]

        mask = np.zeros([shape[0], shape[1], 1], dtype=np.uint8)

        for stream_id,stream in enumerate(streams):

            for trace in stream:
                if trace.stats.channel=="U":
                    start=int(round(trace.stats.sac["a"]*100))
                    end=int(round(trace.stats.sac["t0"]*100))
                else:
                    continue

                mask[stream_id,start:end+1,0]= 1

        class_ids = np.ones([1])

        if self.shuffle:
            random.seed(window_id)
            random_index=random.sample(range(shape[0]),shape[0])
            mask[:,:,0]=mask[:,:,0][random_index]

            streams=[streams[i] for i in random_index]



        station=np.zeros([shape[0],shape[0],2])
        for i,j in itertools.product(range(shape[0]),range(shape[0])):
            station[i,j]=[streams[j][0].stats.sac["stla"]/streams[i][0].stats.sac["stla"],streams[j][0].stats.sac["stlo"]/streams[i][0].stats.sac["stlo"]]


        return mask.astype(np.bool), class_ids.astype(np.int32),station.astype(np.float32)

def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1],mask.shape[0], 2], dtype=np.int32)
    for i in range(mask.shape[-1]):
        # Bounding box.

        for j in range(mask.shape[0]):
            m = mask[j, :, i]
            horizontal_indicies = np.where(m)[0]

            if horizontal_indicies.shape[0]:
                x1, x2 = horizontal_indicies[[0, -1]]

                # x2 should not be part of the box. Increment by 1.
                x2 += 1
            else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
                x1, x2 = 0, 0
            boxes[i,j] = np.array([x1, x2])

    return boxes.astype(np.int32)

def compute_overlap_rate(box, boxes):
    """Calculates overlap rate of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    """
    # Calculate intersection areas

    x1 = np.maximum(box[0], boxes[:, 0])
    x2 = np.minimum(box[1], boxes[:, 1])
    intersection = np.maximum(x2 - x1, 0)
    boxes_area = boxes[:, 1] - boxes[:, 0]

    overlap = intersection/boxes_area

    return overlap


def default(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))

def myconverter(o):
    if isinstance(o, datetime.datetime):
        return o.__str__()
    elif type(o).__module__ == np.__name__:
        return o.__str__()
    else:
        return o

class Evaluation_confidence_mask():
    def __init__(self,single_model,multi_model,dataset,overlap_threshold=0.3):
        self.single_model=single_model
        self.multi_model=multi_model
        self.dataset=dataset
        self.overlap_threshold=overlap_threshold

    def evaluate(self,window_id=None):

        test_results = []
        for window_id in self.dataset.window_ids:


            streams = self.dataset.load_streams(window_id)
            window = self.dataset.load_window(window_id)
            mask,ids,station = self.dataset.load_mask(window_id)

            single_r=self.single_model.detect(np.expand_dims(window,axis=1))
            r=self.multi_model.detect(np.expand_dims(window,axis=0),np.expand_dims(station,axis=0))[0]
            for multi_box_id,box in enumerate(r["rois"]):
                for station_id,single_result in enumerate(single_r):
                    overlap = compute_overlap_rate(box,single_result["rois"])
                    if sum(overlap > self.overlap_threshold) > 0:
                        single_box_id = np.argmax(overlap)
                    else :
                        continue

                    r["match_ids"][multi_box_id][station_id] = single_result["class_ids"][single_box_id]
                    r["match_scores"][multi_box_id][station_id] = single_result["scores"][single_box_id]


                    r["masks"][station_id][:,multi_box_id] = np.squeeze(single_result["masks"],axis=0)[:,single_box_id]


            r["masks"]=extract_bboxes(r["masks"])

            for key,value in r.items():
                r[key]=default(value)

            streams_info={}
            for i,stream in enumerate(streams):
                tr=stream.select(channel="U")[0]
                station=tr.stats.station
                sac_dict=dict(tr.stats.sac)

                for key,value in sac_dict.items():
                    sac_dict[key]=myconverter(value)

                streams_info[station]=sac_dict
                if i ==0:
                    r["starttime"]=myconverter(tr.stats.starttime.datetime)
                    r["endtime"]=myconverter(tr.stats.endtime.datetime)

            r["streams_info"]=streams_info
            r["window_id"]=str(window_id)
            r["event_id"]=self.dataset.window_info[window_id]["main_name"]

            test_results.append(r)
            if int(window_id)%500 ==0:
                print("{}% done.".format(int(window_id)/len(self.dataset.window_ids)))



        return test_results

    def write_json(self,metadata,dir_path):

        json_name="pspicker_meta.json"
        with open(os.path.join(dir_path,json_name),"w") as outfile:
            json.dump(metadata,outfile)


def main():

    single_config=SingleInferenceConfig()
    single_config.display()
    multi_config=MultiInferenceConfig()
    multi_config.display()

    single_model=SingleModel.MaskRCNN(mode="inference", config=single_config,
                                  model_dir=MODEL_DIR)
    single_model.load_weights(SINGLE_MODEL_PATH,by_name=True)
    print("Single station model has been loaded.")

    multi_model=MultiModel.MaskRCNN(mode="inference", config=multi_config,
                                  model_dir=MODEL_DIR)
    multi_model.load_weights(MULTI_MODEL_PATH,by_name=True)
    print("Multi station model has been loaded.")

    with open(TEST_DICT)as f:
        test_dict=json.load(f)

    dataset=PSpickerDataset()
    dataset.load_sac(test_dict,add_sub=False)
    dataset.prepare()

    print("Start evaluation process.")
    evaluation = Evaluation_confidence_mask(single_model,multi_model,dataset,overlap_threshold=0.3)
    results = evaluation.evaluate()

    evaluation.write_json(results,EVAL_DIR)


if __name__ == '__main__':
    main()
