## General utilities
import os.path
import time
import shutil
import sys

# from pydrive2.auth import GoogleAuth
# from pydrive2.drive import GoogleDrive

## Labelbox utilities
import labelbox as lb

# Local functions and definitions
from detectron2.data import MetadataCatalog

import data_utils
from config import *
import config
from data_preperation_in_labelbox import delete_labels_from_queued_datarows, compute_number_of_unlabeled_datarows, \
    download_unlabeled_datarows
from evaluate_performance import inference_preview, val_inference_preview
from inference_labels import delete_files_from_drive, inference_labels
from upload_to_labelbox import upload_to_labelbox
from utils import get_ontology
from data_utils import download_and_split_data, convert_labels_into_detecron2_format, visualize_training_data, \
    calc_weights_for_imbalance_datasets
from train import train_model, set_predictor

if __name__ == '__main__':

    start_time = time.time()

    if os.path.exists('coco_eval'):
        shutil.rmtree('coco_eval')

    client = lb.Client(LB_API_KEY, "https://api.labelbox.com/graphql")
    ## Get labelbox project
    project = client.get_project(PROJECT_ID)

    ## Get ontology
    ontology, thing_classes = get_ontology(PROJECT_ID, client)
    # thing_classes = thing_classes[:-1]
    print('Available classes: ', thing_classes)

    ###### data_utils
    labels, train_labels, val_labels = download_and_split_data(client)
    # train_labels = train_labels[:3]
    # val_labels = val_labels[:3]
    # data_utils.temp(train_labels)
    if IS_TRAINING:
        if ROI_HEADS_NAME == "CoralStandardROIHeads":
            config.BALANCED_WEIGHTS = calc_weights_for_imbalance_datasets(thing_classes, train_labels)
        convert_labels_into_detecron2_format(thing_classes, train_labels, val_labels)
        dataset_dicts, metadata = visualize_training_data()

    # exit from main
    # sys.exit(0)

    ###### train
    cfg = train_model(thing_classes, train=IS_TRAINING)
    predictor = set_predictor(cfg)

    # exit from main
    # sys.exit(0)

    ###### evaluate_performance
    if not HEADLESS_MODE:
        inference_preview(predictor, ontology, thing_classes, dataset_dicts, metadata) #TODO

    # exit from main
    # sys.exit(0)

    ###### data preparation in labelbox
    delete_labels_from_queued_datarows(client)
    all_datarows, all_datarow_ids, datarow_ids_queued = compute_number_of_unlabeled_datarows(client, labels)
    # all_datarows = all_datarows[:2]
    # datarow_ids_queued = datarow_ids_queued[:2]
    # print(datarow_ids_queued)
    # data_row_queued = download_unlabeled_datarows(all_datarows, all_datarow_ids)
    data_row_queued = download_unlabeled_datarows(all_datarows, datarow_ids_queued)
    #
    # ###### inference
    # data_row_queued = data_row_queued[:2]
    metadata = MetadataCatalog.get(DETECTRON_DATASET_TRAINING_NAME)
    predictions = inference_labels(data_row_queued, predictor, ontology, thing_classes, metadata)

    # exit from main
    # sys.exit(0)

    # ###### upload to labelbox
    upload_to_labelbox(project, start_time, predictions)


