##General utilities
import os.path

##Facebook Detectron2 utilities
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

#Local functions and definitions
from config import *
from utils import CocoTrainer
from coral_arch import coral_rpn
from coral_arch import CoralStandardROIHeads

## Train the model. Change the parameters as per your needs.
def train_model(thing_classes, train=True):
    if MODE == 'object-detection':
        model = 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'
    elif MODE == 'segmentation-rle':
        model = 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
    else:
        IOError("MODE should be object-detection or segmentation-rle")


    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model))
    cfg.DATALOADER.NUM_WORKERS = DATALOADER_NUM_WORKERS
    if RESUME_TRAINING:
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0002999")
    else:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
    cfg.SOLVER.IMS_PER_BATCH = IMS_PER_BATCH
    cfg.SOLVER.CHECKPOINT_PERIOD = CHECKPOINT_PERIOD
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = BATCH_SIZE_PER_IMAGE
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
    cfg.MODEL.PROPOSAL_GENERATOR.NAME = PROPOSAL_GENERATOR_NAME
    cfg.MODEL.ROI_HEADS.NAME = ROI_HEADS_NAME
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = ANCHOR_GENERATOR_SIZES
    cfg.MODEL.RPN.IN_FEATURES = RPN_IN_FEATURES

    if MODE == 'segmentation-rle':
        cfg.INPUT.MASK_FORMAT = 'bitmask'

    if not train:
        return cfg

    cfg.TEST.EVAL_PERIOD = EVAL_PERIOD
    cfg.DATASETS.TRAIN = (DETECTRON_DATASET_TRAINING_NAME,)
    cfg.DATASETS.TEST = (DETECTRON_DATASET_VALIDATION_NAME,)
    cfg.SOLVER.BASE_LR = BASE_LR
    cfg.SOLVER.MAX_ITER = MAX_ITER
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = CocoTrainer(cfg)
    trainer.resume_or_load(resume=RESUME_TRAINING)

    print('start training')

    trainer.train()

    return cfg


## Set newly trained model for inference. Make sure to set the appropriate threshold.
def set_predictor(cfg):
    cfg.TEST.DETECTIONS_PER_IMAGE = DETECTIONS_PER_IMAGE
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = PRELABELING_THRESHOLD  # set threshold for this model
    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "weights", "anchors_5_loss_weightedFixedImbalance", "model_0004999.pth")
    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0002999.pth")

    # Create predictor
    predictor = DefaultPredictor(cfg)
    return predictor