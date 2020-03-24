import sys
sys.path.append("/home/aab10867zc/work/aist/pspicker/code")
import config
import utils
import model_multi_confidence_sta_mask_pure as model
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

    # Parse command line arguments
parser = argparse.ArgumentParser(
    description='Train Mask R-CNN on PS picker.')

parser.add_argument('--log_dir', default=None,required=False,
                    metavar="<log dir>",
                    help='Use this log dir instead create a new one.')
parser.add_argument("--num_gpu",default=4,required=False,type=int,
                    metavar="<gpu count>",
                    help="Number of GPUs to use.(default=4)")
parser.add_argument('--lr', default=0.01,required=False,type=float,
                    metavar="<learning rate>",
                    help='Learning rate.(default=0.01)')


parser.add_argument('--lr_decay', type=float, default=0.2, metavar='DC', help='Learning rate decay rate.')

parser.add_argument('--dropout', type=float, default=0.5, metavar='DO', help='Dropout rate.')


parser.add_argument('--initial_stage', type=int, default=0, metavar='N', help='Stage to start training.')

parser.add_argument('--patience', type=int, default=2, metavar='N', help='Stop training stage after #patiences.')

parser.add_argument('--windows_per_gpu',required=False,type=int,
                    default=2,
                    metavar="<windows per gpu>",
                    help='Number of windows to train with on each GPU.')

parser.add_argument('--train_steps',required=False,type=int,
                    default=1000,
                    metavar="<steps per epoch>",
                    help='Number of training steps per epoch.(default=1000)')
parser.add_argument('--val_stpes',required=False,type=int,
                    default=300,
                    metavar="<validtion steps>",
                    help='Number of validation steps to run at the end of every training epoch. (default=300)')

parser.add_argument('--epochs',required=False,type=int,
                    default=100,
                    metavar="<epochs>",
                    help='Number of training epochs. (default=100)')

parser.add_argument('--num_lr_decays', type=int, default=5, metavar='N', help='Number of learning rate decays.')



args = parser.parse_args()

TRAIN_DICT="/home/aab10867zc/work/aist/pspicker/metadata/pspicker_meta_train_2019-07-17.json"
VALIDATION_DICT="/home/aab10867zc/work/aist/pspicker/metadata/pspicker_meta_validation_2019-07-19.json"
MODEL_DIR="/home/aab10867zc/work/aist/pspicker/training_plan"

print("TRAIN_DICT: ", TRAIN_DICT)
print("VALIDATION_DICT: ", VALIDATION_DICT)
print("MODEL_DIR: ",MODEL_DIR)

def main():

    with open(TRAIN_DICT)as f:
        train_dict=json.load(f)
    with open(VALIDATION_DICT)as f:
        validation_dict=json.load(f)


    psconfig=PSConfig()
    psconfig.display()

    train_dataset=PSpickerDataset()
    train_dataset.load_sac(train_dict,[10,12000,3],add_sub=False)
    train_dataset.prepare()


    validation_dataset=PSpickerDataset()
    validation_dataset.load_sac(validation_dict,[10,12000,3],add_sub=False)
    validation_dataset.prepare()


            # Data generators
    train_generator = model.data_generator(train_dataset, psconfig, shuffle=True,
                                         batch_size=psconfig.BATCH_SIZE)
    val_generator = model.data_generator(validation_dataset, psconfig, shuffle=True,
                                       batch_size=psconfig.BATCH_SIZE)

    now = datetime.datetime.now()
    if args.log_dir:
        log_dir=args.log_dir
    else:
        log_dir = os.path.join(MODEL_DIR, "{}{:%Y%m%dT%H%M%S}".format(
        psconfig.NAME.lower(), now))
        os.makedirs(log_dir, exist_ok=True)

    for stage in range(args.initial_stage, args.num_lr_decays):

        stage_train_dir = make_path(log_dir, stage)
        previous_stage_train_dirs = [make_path(log_dir, stage) for stage in range(0, stage)]
        next_stage_train_dir = make_path(log_dir, stage + 1)

    # Pass if there is a training directory of the next stage.
        if os.path.isdir(next_stage_train_dir):
            continue

    # Setup the initial learning rate for the stage.
        lr = args.lr * (args.lr_decay ** stage)

    # Create a directory for the stage.
        os.makedirs(stage_train_dir, exist_ok=True)

    # Find the best checkpoint in the current stage.
        ckpt_path, ckpt_epoch, ckpt_val_loss = find_best_checkpoint(stage_train_dir)
    # If there is no checkpoint in the current stage, then find it in the previous stage.
        if ckpt_path is None:
            ckpt_path, ckpt_epoch, ckpt_val_loss = find_best_checkpoint(*previous_stage_train_dirs[-1:])

        print('\n=> Start training stage {}: lr={}, train_dir={}'.format(stage, lr, stage_train_dir))
        if ckpt_path:
            print('=> Found a trained model: epoch={}, val_loss={}, path={}'.format(ckpt_epoch, ckpt_val_loss, ckpt_path))
        else:
            print('=> No trained model found.')


        psmodel = MaskRCNN_refined(mode="training", config=psconfig,model_dir=stage_train_dir)
        psmodel.set_log_dir(ckpt_path)

        csv_logger = keras.callbacks.CSVLogger(make_path(stage_train_dir, 'training.csv'), append=True)
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.patience)

        psmodel.train(train_generator,val_generator,lr,ckpt_epoch + 1,args.epochs,custom_callbacks=[csv_logger,early_stopping])

        best_ckpt_path, *_ = find_best_checkpoint(stage_train_dir)
        print('=> The end of the stage. Start evaluation on test set using checkpoint "{}".'.format(best_ckpt_path))
    print('\n=> Done.\n')

class MaskRCNN_refined(model.MaskRCNN):


    def train(self, train_generator, val_generator, learning_rate, initial_epoch,epochs,
              custom_callbacks=None):
        """Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs. Note that previous training epochs
                are considered to be done alreay, so this actually determines
                the epochs to train in total rather than in this particaular
                call.
        layers: Allows selecting wich layers to train. It can be:
            - A regular expression to match layer names to train
            - One of these predefined values:
              heads: The RPN, classifier and mask heads of the network
              all: All the layers
              3+: Train Resnet stage 3 and up
              4+: Train Resnet stage 4 and up
              5+: Train Resnet stage 5 and up

                ])
        custom_callbacks: Optional. Add custom callbacks to be called
        with the keras fit_generator method. Must be list of type keras.callbacks.
        no_augmentation_sources: Optional. List of sources to exclude for
            augmentation. A source is string that identifies a dataset and is
            defined in the Dataset class.
        """
        assert self.mode == "training", "Create model in training mode."


        if self.checkpoint_path:
            print('=> Load weights from "{}".'.format(self.checkpoint_path))
            self.load_weights(self.checkpoint_path,by_name=True)


        # Callbacks
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(os.path.join(self.log_dir, "ckpt-e{epoch:03d}-l{val_loss:.4f}.h5"),
                                            verbose=0, save_weights_only=True,save_best_only=True),
        ]

        # Add custom callbacks to the list
        if custom_callbacks:
            callbacks +=custom_callbacks

        # Train


        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

        # Work-around for Windows: Keras fails on Windows when using
        # multiprocessing workers. See discussion here:
        # https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
        if os.name is 'nt':
            workers = 0
        else:
            workers = multiprocessing.cpu_count()

        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=initial_epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=self.config.VALIDATION_STEPS,
            max_queue_size=100,
            workers=workers,
            use_multiprocessing=True,
        )


    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.
        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """

        # Directory for training logs
        self.log_dir = os.path.join(self.model_dir)

        self.checkpoint_path=model_path


def find_best_checkpoint(*dirs, prefix='ckpt'):
    best_checkpoint_path = None
    best_epoch = -1
    best_val_loss = 1e+10
    for dir in dirs:
        checkpoint_paths = glob('{}/{}*'.format(dir, prefix))
        for checkpoint_path in checkpoint_paths:
            epoch = int(re.findall('e\d+', checkpoint_path)[0][1:])
            val_loss = float(re.findall('l\d\.\d+', checkpoint_path)[0][1:])

            if val_loss < best_val_loss:
                best_checkpoint_path = checkpoint_path
                best_epoch = epoch
                best_val_loss = val_loss

    return best_checkpoint_path, best_epoch, best_val_loss

def make_path(*paths):
    path=os.path.join(*[str(path) for path in paths])
    path=os.path.realpath(path)
    return path



#neighbour stations
#no substations

class PSpickerDataset(model.Dataset):
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

        random.seed(620)

        # Add windows
        len_dataset=len(sac_info["windows"])
        for window_id,main_event in enumerate(sac_info["windows"]):


            sub_event=sac_info["windows"][random.choice([sub_id for sub_id in range(len_dataset) if sub_id!=window_id])]

            if add_sub:
                num_main_event_least=shape[0]//2
            else:
                num_main_event_least=shape[0]

            num_main_event=min(random.choice(range(num_main_event_least,shape[0]+1)),int(main_event["num_stations"]))
            main_stations=random.sample(main_event["stations"],num_main_event)
            main_paths=[main_event["traces"][station] for station in main_stations]


            sub_stations_total=[station for station in sub_event["stations"] if station not in main_event["stations"]]
            sub_stations=random.sample(sub_stations_total,min(shape[0]-num_main_event,len(sub_stations_total)))
            num_sub_event=len(sub_stations)

            sub_paths=[sub_event["traces"][station] for station in sub_stations]

            path=main_paths+sub_paths

            if len(path)<shape[0]:
                continue

            main_match=np.zeros(shape[0],dtype=np.int32)
            main_match[:num_main_event]=1

            sub_match=np.zeros(shape[0],dtype=np.int32)
            sub_match[num_main_event:num_main_event+num_sub_event]=1

            self.add_window("pspicker",window_id=window_id,main_stations=main_stations,sub_stations=sub_stations,
                            main_name=main_event["name"],sub_name=sub_event["name"],main_match=main_match,sub_match=sub_match,shape=shape,path=path)



    def load_streams(self,window_id):
        info = self.window_info[window_id]
        shape=info["shape"]
        streams=[]

        for event in info["path"]:
            paths=list(event.values())
            traces=[]
            for path in paths:
                trace=read(path)[0]
                traces.append(trace)

            stream=Stream(traces=traces)
            stream.detrend("constant")
            stream.filter("highpass", freq=2.0)

            for i in range(len(stream)):
                stream[i].data-=np.mean(stream[i].data)
                stream[i].data/=np.std(stream[i].data)

            streams.append(stream)

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
        main_match=info["main_match"]
        sub_match=info["sub_match"]


        if len(info["sub_stations"])>=3:
            count=2
        elif len(info["sub_stations"])<3:
            count=1



        mask = np.zeros([shape[0], shape[1], count], dtype=np.uint8)

        for stream_id,stream in enumerate(streams):

            for trace in stream:
                if trace.stats.channel=="U":
                    start=int(round(trace.stats.sac["a"]*100))
                    end=int(round(trace.stats.sac["t0"]*100))
                else:
                    continue
            if stream_id in np.where(main_match)[0]:

                mask[stream_id,start:end+1,0]= 1
            elif stream_id in np.where(sub_match)[0] and count==2:
                mask[stream_id,start:end+1,1]= 1

        class_ids = np.ones([count])

        if self.shuffle:
            random.seed(window_id)
            random_index=random.sample(range(shape[0]),shape[0])
            mask[:,:,0]=mask[:,:,0][random_index]
            main_match=main_match[random_index]
            streams=[streams[i] for i in random_index]

            if count==2:
                mask[:,:,1]=mask[:,:,1][random_index]
                sub_match=sub_match[random_index]

        if count==2:
            match=np.concatenate((np.expand_dims(main_match,axis=0),np.expand_dims(sub_match,axis=0)),axis=0)
        elif count==1:
            match=np.expand_dims(main_match,axis=0)



        identity=np.zeros([shape[0],shape[0],1])
        main_ids=np.where(main_match)[0]
        sub_ids=np.where(sub_match)[0]

        identity[main_ids]=np.expand_dims(main_match,axis=1)
        identity[sub_ids]=np.expand_dims(sub_match,axis=1)

        station=np.zeros([shape[0],shape[0],2])
        for i,j in itertools.product(range(shape[0]),range(shape[0])):
            station[i,j]=[streams[j][0].stats.sac["stla"]/streams[i][0].stats.sac["stla"],streams[j][0].stats.sac["stlo"]/streams[i][0].stats.sac["stlo"]]


        return mask.astype(np.bool), class_ids.astype(np.int32),match.astype(np.int32),station.astype(np.float32),identity.astype(np.int32)



class PSConfig(config.Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "pspicker"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    WINDOWS_PER_GPU = args.windows_per_gpu

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    GPU_COUNT=args.num_gpu

    LEARNING_RATE=args.lr

    STEPS_PER_EPOCH=args.train_steps

    VALIDATION_STEPS=args.val_stpes

    DROP_RATE=args.dropout

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
    MASK_CONV=False


if __name__ == '__main__':
    main()
