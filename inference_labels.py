##General utilities
from uuid import uuid4
import os, os.path
import cv2
import time
import progressbar
import json

#Local functions and definitions
from config import *
from evaluate_performance import val_inference_preview
from utils import mask_to_cloud, get_orientation, transfer_mask_based_on_orientation

#new for mask to polygon
import rasterio
from rasterio import features
import shapely
from shapely.geometry import Point, Polygon
import numpy as np



def convert_mask_to_correct_format_polygon(pred_mask):
    polygons = mask_to_polygons_layer(pred_mask)
    if polygons.type != 'Polygon':
        return None
    polygons_in_2_array = list(zip(*polygons.exterior.xy))
    return [dict(zip(['x', 'y'], l)) for l in polygons_in_2_array]

def mask_to_polygons_layer(mask):  # TODO Amit some lines of code are unreachable
    all_polygons = []
    for shape, value in features.shapes(mask.astype(np.int16), mask=(mask >0), transform=rasterio.Affine(1.0, 0, 0, 0, 1.0, 0)):
        return shapely.geometry.shape(shape).simplify(SIMPLIFY_VALUE_POLYGON, preserve_topology=True)  # TODO Amit tolerance value is just for the code to run, need to add from config or smarter way

from labelbox.data.annotation_types import (
    Label, ImageData, MaskData, LabelList, TextData, VideoData,
    ObjectAnnotation, ClassificationAnnotation, Polygon, Rectangle, Line, Mask,
    Point, Checklist, Radio, Text, TextEntity, ClassificationAnswer)

def delete_files_from_drive(drive):
    file_list = drive.ListFile({'q': "'{}' in parents and trashed=false".format(GOOGLE_DRIVE_ID)}).GetList()
    for file in file_list:
        file.Delete()

def create_obj_dict(polygon, classname, score):
    return {'polygon': polygon, 'group': classname, 'confidence': score}

def create_img_dict(external_id, img_class=None):
    return {'External ID': external_id, 'Objects': [], 'Img Class': img_class}

def inference_labels(data_row_queued, predictor, ontology, thing_classes, metadata):

    predictions = []
    predictions_for_agisoft = []

    print("Inferencing...\n")
    time.sleep(1)
    bar = progressbar.ProgressBar(maxval=len(data_row_queued), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    for datarow_idx, datarow in enumerate(data_row_queued):
        extension = os.path.splitext(datarow.external_id)[1]
        filename = str(DATA_LOCATION/inference/datarow.uid) + extension
        im = cv2.imread(filename)
        orientation = get_orientation(filename)

        ##Predict using FB Detectron2 predictor
        outputs = predictor(im)

        if SHOW_ANNOTATIONS:
            val_inference_preview(im, metadata, outputs)

        categories = outputs["instances"].to("cpu").pred_classes.numpy()
        predicted_boxes = outputs["instances"].to("cpu").pred_boxes
        scores = outputs["instances"].to("cpu").scores.numpy().astype(float)

        if INFERENCE_TO_AGISOFT:
            predictions_for_agisoft.append(create_img_dict(datarow.external_id))

        for i, category in enumerate(categories):
            # if i > 5:
            #     break
            classname = thing_classes[category]

            for item in ontology:
                if classname == item['name']:
                    schema_id = item['shape_featureSchemaId']
                    if (classname != 'Mouth') and HIRERARCHY:
                        question = item['parent_featureSchemaId']
                        answer   = item['featureSchemaId']

            if MODE == 'segmentation-rle':
                pred_mask = outputs["instances"][i].to("cpu").pred_masks.numpy()

                # polygon_in_correct_format = convert_mask_to_correct_format_polygon(pred_mask)
                polygon_in_correct_format = convert_mask_to_correct_format_polygon(
                    transfer_mask_based_on_orientation(pred_mask.squeeze(), orientation, 'inference'))
                if polygon_in_correct_format is None:
                    continue

                if INFERENCE_TO_AGISOFT:
                    predictions_for_agisoft[datarow_idx]['Objects'].append(create_obj_dict(polygon_in_correct_format, classname, scores[i]))

                if classname == 'Mouth':
                    predictions.append({"uuid": str(uuid4()),
                                        'schemaId': schema_id,
                                        'point': {"x": 30, "y": 150}, #TODO fake, need to fix mouth
                                        'dataRow': {'id': datarow.uid}})
                else:
                    predictions.append({"uuid": str(uuid4()),
                                        'schemaId': schema_id,
                                        'polygon': polygon_in_correct_format,  #cl09kxrrq4vi60zaeas5mee9j
                                        'dataRow': {'id': datarow.uid},
                      # 'classifications': [#for hirarechy
                      #     {"schemaId": question, #phase schema id - type radio
                      #      "answer": {"schemaId": answer} # Nested checklist answer
                      #      }]
                                        })

            if MODE == 'object-detection':
                bbox = predicted_boxes[i].tensor.numpy()[0]
                bbox_dimensions = {'left': int(bbox[0]), 'top': int(bbox[1]), 'width': int(bbox[2] - bbox[0]),
                                   'height': int(bbox[3] - bbox[1])}
                predictions.append({"uuid": str(uuid4()), 'schemaId': schema_id, 'bbox': bbox_dimensions,
                                    'dataRow': {'id': datarow.uid}})

        # print('\predicted '+ str(datarow_idx) + ' of ' + str(len(data_row_queued)))
        bar.update(datarow_idx)

    bar.finish()
    time.sleep(1)
    print('Total annotations predicted: ', len(predictions))

    if INFERENCE_TO_AGISOFT:
        (DATA_LOCATION / OUTPUT_PATH).parent.mkdir(exist_ok=True)
        with open(str(DATA_LOCATION/OUTPUT_PATH), 'w') as f:
            json.dump(predictions_for_agisoft, f, ensure_ascii=False)

    return predictions

