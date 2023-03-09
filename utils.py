##General utilities
import datetime as dt
import pathlib
from uuid import uuid4

import PIL
import pickle
import requests
import os, os.path
import numpy as np
import cv2
from skimage import io
import simplejson as json
import time
from matplotlib import pyplot as plt
from pycocotools import mask
import progressbar
from PIL import Image, ExifTags

##Facebook Detectron2 utilities
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

# Local functions and definitions
from config import *
from coral_arch.CoralDefaultTrainer import CoralDefaultTrainer


# Swap function
def change_idx_position(l, pos1, pos2):
    elem = l.pop(pos1)
    l.insert(pos2, elem)
    return l

## get project ontology from labelbox
def get_ontology(project_id, client):
    response = client.execute(
        """
        query getOntology (
            $project_id : ID!){ 
            project (where: { id: $project_id }) { 
                ontology { 
                    normalized 
                } 
            }
        }
        """,
        {"project_id": project_id})


    ontology = response['project']['ontology']['normalized']['tools']

    if CHANGE_ORDER:
        ontology = change_idx_position(ontology, 0, 3)

    ##Return list of tools and embed category id to be used to map classname during training and inference
    mapped_ontology = []
    thing_classes = []

    i = 0
    for item in ontology:
        #         if item['tool']=='superpixel' or item['tool']=='rectangle':
        if SKIP_MOUTHS and item['name'] == 'Mouth':
            continue
        if len(item['classifications']) == 0:
            item.update({'category': i})
            item.update({'shape_featureSchemaId': item['featureSchemaId']})
            mapped_ontology.append(item)
            thing_classes.append(item['name'])
            i = i + 1
        else:
            for classification in item['classifications']:
                for option in classification['options']:
                    option.update({'category': i})
                    option.update({'name': option['label']})
                    option.update({'shape_featureSchemaId': item['featureSchemaId']})
                    option.update({'parent_featureSchemaId': classification['featureSchemaId']})

                    mapped_ontology.append(option)
                    thing_classes.append(option['label'])
                    i = i + 1

    # for item in ontology:
    #     #         if item['tool']=='superpixel' or item['tool']=='rectangle':
    #     item.update({'category': i})
    #     mapped_ontology.append(item)
    #     thing_classes.append(item['name'])
    #     i = i + 1

    return mapped_ontology, thing_classes


## Creates a new export request to get all labels from labelbox.
def get_labels(project_id, client):
    should_poll = 1
    while should_poll == 1:
        response = client.execute(
            """
            mutation export(
            $project_id : ID!    
            )
            { 
                exportLabels(data:{ projectId: $project_id }){ 
                    downloadUrl 
                    createdAt 
                    shouldPoll 
                }
            }
            """,
            {"project_id": project_id})

        if not response['exportLabels']['shouldPoll']:
            should_poll = 0
            url = response['exportLabels']['downloadUrl']
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36"}

            r = requests.get(url, headers=headers)

            print('Export generated')
            ## writing export to disc for easier debugging
            open('export.json', 'wb').write(r.content)
            return r.content
        else:
            print('Waiting for export generation. Will check back in 10 seconds.')
            time.sleep(10)

    return response


## Get all previous predictions import (bulk import request).
def get_current_import_requests(client):
    response = client.execute(
        """
        query get_all_import_requests(
            $project_id : ID! 
        ) {
          bulkImportRequests(where: {projectId: $project_id}) {
            id
            name
          }
        }
        """,
        {"project_id": PROJECT_ID})

    return response['bulkImportRequests']


## Delete all current predictions in a project and dataset. We want to delete them and start fresh with predictions from the latest model iteration
def delete_import_request(import_request_id, client):
    response = client.execute(
        """
            mutation delete_import_request(
                $import_request_id : ID! 
            ){
              deleteBulkImportRequest(where: {id: $import_request_id}) {
                id
                name
              }
            }
        """,
        {"import_request_id": import_request_id})

    return response


def diff_lists_new(li1, li2):
    li_dif = [i for i in li1 if i not in li2]
    return li_dif

## function to return the difference between two lists. This is used to compute the queued datarows to be used for inference.
def diff_lists(li1, li2):
    li_dif = [i for i in li1 + li2 if i not in li1 or i not in li2]
    return li_dif


## Generic data download function
def download_files(filemap, force_download=False):
    path, uri = filemap
    ## Download data
    if (not os.path.exists(path)) or force_download:
        r = requests.get(uri, stream=True)
        if r.status_code == 200:
            with open(path, 'wb') as f:
                for chunk in r:
                    f.write(chunk)
    return path


## Converts binary image mask into COCO RLE format
def rle_encode(mask_image):
    size = list(mask_image.shape)
    pixels = mask_image.flatten()

    # We avoid issues with '1' at the start or end (at the corners of
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask,
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]

    rle = {'counts': runs.tolist(), 'size': size}
    return rle


def load_set(dir):
    with open(dir + "dataset.json") as json_file:
        dataset_dicts = json.loads(json_file)
    return dataset_dicts


def cv2_imshow(a, **kwargs):
    #     a = a.clip(0, 255).astype('uint8')
    # cv2 stores colors as BGR; convert to RGB
    if a.ndim == 3:
        if a.shape[2] == 4:
            a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
        else:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)

    return plt.imshow(a, **kwargs)


# def upload_to_gcs(file_name, storage_client):
#     bucket = storage_client.get_bucket("predictions-import-test")
#     blob = bucket.blob("{}.png".format(str(uuid4())))
#     blob.upload_from_filename(file_name)
#     return blob.generate_signed_url(dt.timedelta(weeks=10))


def mask_to_cloud(img, mask_array, filename):
    num_instances = mask_array.shape[0]
    mask_array = np.moveaxis(mask_array, 0, -1)
    mask_array_instance = []
    output = np.zeros_like(img)
    for i in range(num_instances):
        mask_array_instance.append(mask_array[:, :, i:(i + 1)])
        output = np.where(mask_array_instance[i] == True, 255, output)
    im = Image.fromarray(output)
    path = str(DATA_LOCATION / tmp / (filename + '.png'))  # new instead of upload_to_gcs
    im.save(path)

    # cloud_mask = upload_to_gcs(DATA_LOCATION + 'tmp/' + filename + '.png', storage_client)

    return path

def transfer_mask_based_on_orientation(im_mask, orientation, stage):
    if orientation == 1:
        return im_mask
    elif (stage == 'training' and orientation == 6) or (stage == 'inference' and orientation == 8):
        return np.rot90(im_mask, 3)
    elif (stage == 'training' and orientation == 8) or (stage == 'inference' and orientation == 6):
        return np.rot90(im_mask)

def get_orientation(filename):
    pil_image = PIL.Image.open(filename)
    exif = pil_image._getexif()
    return exif[orientation_tag]

orientation_tag = [key for key, val in ExifTags.TAGS.items() if ExifTags.TAGS[key] == 'Orientation'][0]



## Convert and load labelbox labels into Detectron2 format
def load_detectron2_dataset(labels, thing_classes, dir, existing_json_path=None):
    if existing_json_path:
        # f = open(DATA_LOCATION / existing_json_path, "r")
        # dataset_dicts = json.load(f)
        # f.close()
        # return dataset_dicts
        with open(DATA_LOCATION / existing_json_path, 'rb') as f:
            return pickle.load(f)

    dataset_dicts = []
    j = 0
    total = len(labels)

    print("Num labels processing: " + str(total))
    time.sleep(1)
    bar = progressbar.ProgressBar(maxval=total, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    for label in labels:

        try:
            record = {}
            filename = os.path.join(dir, label['External ID'])
            print(label['External ID'])
            ##scikit needed to raise exception if unable to read the image
            _ = io.imread(filename)

            im = cv2.imread(filename)
            height, width = im.shape[:2]

            # pil_image = PIL.Image.open(filename)
            # exif = pil_image._getexif()
            # orientation = exif[orientation_tag]

            orientation = get_orientation(filename)

            record["file_name"] = filename
            record["height"] = height
            record["width"] = width
            record["image_id"] = label['ID']

            objs = []

            # for i, instance in enumerate(label['Label']['objects']):
            for i, instance in enumerate(label['Label']['objects'][::-1]):
                if SKIP_MOUTHS and instance['title'] == 'Mouth':
                    continue
                # if i > 1:
                #     break
                # category_id = thing_classes.index(instance['title'])
                #                 print(category_id)

                if 'classifications' not in instance:
                    category_id = thing_classes.index(instance['title'])
                else:
                    category_id = thing_classes.index(instance['classifications'][0]['answer']['title'])



                if MODE == 'object-detection':
                    obj = {
                        "bbox": [instance['bbox']['left'], instance['bbox']['top'], instance['bbox']['width'],
                                 instance['bbox']['height']],
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "segmentation": [],
                        "category_id": category_id,
                    }
                    objs.append(obj)

                if MODE == 'segmentation-rle':
                    path = DATA_LOCATION / masks / label['External ID']
                    path = path.parent / (path.stem + '.png')
                    path = str(path)
                    mask_URI = instance['instanceURI']
                    downloaded_path = download_files((path, mask_URI), force_download=True)
                    im_mask = cv2.imread(downloaded_path, 0)

                    im_mask = transfer_mask_based_on_orientation(im_mask, orientation, 'training')

                    # im_vis = im
                    # im_vis += (0.3 * im_mask[..., None]).astype(np.uint8)
                    # im_vis[im_mask > 0] = 0.7 * im_mask[im_mask > 0] + 0.3 * im[im_mask > 0]
                    # if orientation == 6:
                    #     plt.imshow(im_vis)
                    #     plt.show()

                    binary = np.array(im_mask)

                    rle = mask.encode(np.asfortranarray(binary))
                    ground_truth_bounding_box = mask.toBbox(rle)

                    obj = {
                        "bbox": ground_truth_bounding_box.tolist(),
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "segmentation": rle,
                        "category_id": category_id,
                        "iscrowd": 0
                    }
                    objs.append(obj)

            record["annotations"] = objs
            dataset_dicts.append(record)

            bar.update(j + 1)
            j = j + 1
        except Exception as e:
            print('Exception: ', e)

    bar.finish()

    ## Write detectron2 dataset file to disk for easier debugging
    f = open(dir + "dataset_dict.json", "w")
    f.write(json.dumps(dataset_dicts))
    f.close()

    with open(dir + "dataset_dict.pickle", 'wb') as f:
        pickle.dump(dataset_dicts, f)

    return dataset_dicts


class CocoTrainer(CoralDefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs("coco_eval", exist_ok=True)
            output_folder = "coco_eval"

        return COCOEvaluator(dataset_name, cfg, False, output_folder)
