##General utilities
import cv2
import random
from matplotlib import pyplot as plt

##Facebook Detectron2 utilities
from detectron2.utils.visualizer import Visualizer

#Local functions and definitions
from config import *


###### preview inference
def inference_preview(predictor, ontology, thing_classes, dataset_dicts, metadata):
    # Let's perform inferencing on random samples and preview the predictions before proceeding.
        for d in random.sample(dataset_dicts, 3):
            im = cv2.imread(d["file_name"])
            outputs = predictor(im)
            categories = outputs["instances"].to("cpu").pred_classes.numpy()
            predicted_boxes = outputs["instances"].to("cpu").pred_boxes

            if MODE == 'segmentation-rle':
                pred_masks = outputs["instances"].to("cpu").pred_masks.numpy()

            if len(categories) != 0:
                for i in range(len(categories)):
                    classname = thing_classes[categories[i]]
                    for item in ontology:
                        if classname == item['name']:
                            schema_id = item['featureSchemaId']

            v = Visualizer(im[:, :, ::-1],
                           metadata=metadata,
                           scale=2,
                           )
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

            ## For paperspace cloud notebooks. Cloud notebooks do not support cv2.imshow.
            plt.rcParams['figure.figsize'] = (12, 24)
            plt.imshow(v.get_image()[:, :, ::-1])
            plt.show()


def val_inference_preview(im, metadata, outputs):
    # Let's perform inferencing on random samples and preview the predictions before proceeding.


    v = Visualizer(im[:, :, ::-1],
                           metadata=metadata,
                           scale=2,
                           )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    ## For paperspace cloud notebooks. Cloud notebooks do not support cv2.imshow.
    plt.rcParams['figure.figsize'] = (12, 24)
    plt.imshow(v.get_image()[:, :, ::-1])
    plt.show()

