
##General utilities
import pickle

import cv2
from matplotlib import pyplot as plt
import os.path
import simplejson as json
from multiprocessing.pool import ThreadPool
import random

##Facebook Detectron2 utilities
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer

#Local functions and definitions
from config import *
from utils import get_labels, download_files, load_detectron2_dataset


# The function downloads data from labelbox (if not already exist on the computer) and splits the data into training
# and validation sets

def temp(train_labels):
    for img in train_labels:
        x_dist = []
        y_dist = []
        for annotation in img['Label']['objects']:
            pointsx = [pnt['x'] for pnt in annotation['polygon']]
            pointsy = [pnt['y'] for pnt in annotation['polygon']]
            x_dist.append(max(pointsx) - min(pointsx))
            y_dist.append(max(pointsy) - min(pointsy))
        print('max x for img:', max(x_dist), 'min x for img:', min(x_dist))
        print('max y for img:', max(y_dist), 'min y for img:', min(y_dist))


def calc_weights_for_imbalance_datasets(thing_classes, train_labels):
    class_hist = dict.fromkeys(thing_classes + ['Background class'], 0)
    for img in train_labels:
        for annotation in img['Label']['objects']:
            group = annotation['classifications'][0]['answer']['title']
            class_hist[group] = class_hist[group] + 1
    sum_of_annotations = sum(class_hist.values())
    class_hist['Background class'] = (1 / 5) * sum_of_annotations
    sum_of_annotations = sum_of_annotations + class_hist['Background class']
    balanced_weights_unormalized =  {key: sum_of_annotations / value for key, value in class_hist.items()}
    balanced_weights = {key: value / sum(balanced_weights_unormalized.values()) for key, value in balanced_weights_unormalized.items()}
    print('Balanced weights:')
    for k, v in balanced_weights.items():
        print(k, v)
    return balanced_weights
    # with open(DATA_LOCATION / WEIGHTS_UNBALANCED_DATA, "wb") as fp:  # Pickling
    #     pickle.dump(balanced_weights, fp)


def download_and_split_data(client):

    ##Get labels
    get_labels(PROJECT_ID, client)
    labels = json.loads(get_labels(PROJECT_ID, client))

    ##Split training and validation labels
    if NUM_SAMPLE_LABELS != 0:
        val_sample = int(VALIDATION_RATIO * NUM_SAMPLE_LABELS)
        val_labels = random.sample(labels, val_sample)
        train_labels = random.sample(labels, NUM_SAMPLE_LABELS)
    else:
        split = int(VALIDATION_RATIO * len(labels))
        val_labels = labels[:split]
        train_labels = labels[split:]

    ## Check and create folders for downloading data from Labelbox #TODO move to config?
    DATA_LOCATION.mkdir(exist_ok=True, parents=True)
    (DATA_LOCATION/train).mkdir(exist_ok=True)
    (DATA_LOCATION/val).mkdir(exist_ok=True)
    (DATA_LOCATION/inference).mkdir(exist_ok=True)
    (DATA_LOCATION/masks).mkdir(exist_ok=True)
    (DATA_LOCATION/tmp).mkdir(exist_ok=True)

    ##Download training and validation labels in parallel
    train_urls = []
    for label in train_labels:
        train_urls.append((str(DATA_LOCATION/train/label['External ID']), label['Labeled Data']))

    val_urls = []
    for label in val_labels:
        val_urls.append((DATA_LOCATION/val/label['External ID'], label['Labeled Data']))

    if DOWNLOAD_IMAGES:
        print('Downloading training and validation data... \n')

        results_train = ThreadPool(NUM_CPU_THREADS).imap_unordered(download_files, train_urls)
        results_val = ThreadPool(NUM_CPU_THREADS).imap_unordered(download_files, val_urls)

        for item in results_train:
            pass
        for item in results_val:
            pass

        print('Finished downloading training and validation data... \n')

    return labels, train_labels, val_labels


# The function converts the labalbox labeling format into the model labeling format
def convert_labels_into_detecron2_format(thing_classes, train_labels, val_labels):
    ### Begin FB Detectron code.
    #tmp
    # a = load_detectron2_dataset(train_labels, thing_classes,
    #                         str(DATA_LOCATION / train), existing_json_path=EXISTING_JSON_VAL_PATH)
    # Load dataset into Detectron2
    try:
        DatasetCatalog.register(DETECTRON_DATASET_TRAINING_NAME,
                                lambda: load_detectron2_dataset(train_labels, thing_classes,
                                                                str(DATA_LOCATION/train), existing_json_path=EXISTING_JSON_TRAINING_PATH))
        DatasetCatalog.register(DETECTRON_DATASET_VALIDATION_NAME,
                                lambda: load_detectron2_dataset(val_labels, thing_classes,
                                                                str(DATA_LOCATION/val), existing_json_path=EXISTING_JSON_VAL_PATH))
        MetadataCatalog.get(DETECTRON_DATASET_TRAINING_NAME).thing_classes = thing_classes
        MetadataCatalog.get(DETECTRON_DATASET_VALIDATION_NAME).thing_classes = thing_classes
    except Exception as e:
        print(e)


def visualize_training_data():

    ##Load data and metadata for visualization and inference
    dataset_dicts = DatasetCatalog.get(DETECTRON_DATASET_TRAINING_NAME)
    dataset_dicts_val = DatasetCatalog.get(DETECTRON_DATASET_VALIDATION_NAME)
    metadata = MetadataCatalog.get(DETECTRON_DATASET_TRAINING_NAME)

    ##check if the training data is loaded correctly
    if not HEADLESS_MODE:
        for d in random.sample(dataset_dicts, 3):
            img = cv2.imread(d["file_name"])
            visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=4)
            vis = visualizer.draw_dataset_dict(d)

            ## For paperspace cloud notebooks. Cloud notebooks do not support cv2.imshow. #TODO change to imshow?
            plt.rcParams['figure.figsize'] = (6, 12)
            plt.imshow(vis.get_image()[:, :, ::-1])
            plt.show()

    return dataset_dicts, metadata
