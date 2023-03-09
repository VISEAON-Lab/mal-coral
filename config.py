import pathlib

MATAN_API = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjanRzN2M2YjA1MmI5MDg3OGYwMmtjMjg0Iiwib3JnYW5pemF0aW9uSWQiOiJjanRzN2M2YWw1aHMzMDc5OXFnOTkwMzl2IiwiYXBpS2V5SWQiOiJjbDR3ZTUycXoydDhxMDd4bTNmbm1iY3lyIiwic2VjcmV0IjoiZWI2ZmRhNGQ4NWNmYmU2MjAzYmRkYmM5OTYzZjVkNGYiLCJpYXQiOjE2NTYzMTMyNDQsImV4cCI6MjI4NzQ2NTI0NH0.W7szDFavgI2DZscmCMe3UR1k5fKX4NYXlhvupDM_mkg"
# AMIT_API = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbDNpcW4wdHExdjh3MDc5MjlhdHJnd21xIiwib3JnYW5pemF0aW9uSWQiOiJjbDNpcW4wdGUxdjh2MDc5MjZvbDBidjdnIiwiYXBpS2V5SWQiOiJjbDRvNDB0enEyODl5MDdiODczenAzZm1iIiwic2VjcmV0IjoiM2RiNTE1MDQ0ZGIyMWZiN2M5YTg5N2NmYjA3YzM5MDYiLCJpYXQiOjE2NTU4MTI1MjAsImV4cCI6MjI4Njk2NDUyMH0.T3SpiCBabIP1rSu1yJHeA9yq77FOtB-yXl0eU2WLtuw"
LB_API_KEY = MATAN_API

MATAN_PROJECT_ID            = 'cl06bz2uhlkyj0zbhdn4ygg5p'
AMIT_PROJECT_ID             = 'cl583jbmzk7d5070e358e8np7'
NEW_NO_HIRERARCHY           = 'cl6ouaz571k9w073j4f18a7e4'
LABELS_FOR_PAPER_HIRERARCHY = 'clbgastd206fh071hgl5a1wlh'
PROJECT_ID = NEW_NO_HIRERARCHY

#some projects have hirerachy and some don't
HIRERARCHY = True
CHANGE_ORDER = False
if PROJECT_ID == NEW_NO_HIRERARCHY:
    HIRERARCHY   = False
    CHANGE_ORDER = True

# DATASETS = ['cl61x75vqh5yj08030a7383ir', 'cl0i77zs2gwhz0zbk6o7wee7g', 'cl0i67hplgnve0zbk1oydhbj5', 'cl09kttll4ukm0zae3984aykf', 'cl09kn3e05qhz0z79cfk44obv', 'cl06f0e9jm7vz0zbhganl7t0v'] #Matan labelbox dataset ids attached to the project
# DATASETS = ['cl0i77zs2gwhz0zbk6o7wee7g', 'cl0i6fncubzve10aldsnrclfg', 'cl0i67hplgnve0zbk1oydhbj5']
DATASETS = ['cl0i77zs2gwhz0zbk6o7wee7g']
# DATASETS = ['cl64z9uurcuml07y49rqi9xee', 'cl6c2pv5t07p408wnee0ogve2', 'cl63o2flu1ili07x5cqdu3une']
# DATASETS = ['clcm28u635kgk07y3byd0adq1', 'clcm23dad5gcj07w4h8e4fzer', 'clcm1kto75h2a08xx5mf24ytc', 'clcm0fb8j5chc08y4eab34j8d', 'clcm049vi5itq07283sq2cake', 'clclzsfez5bf308y44fvi8sma', 'clclz0svf5chz08xxdutd07ev', 'clclyullk4lj608zbhjz176ny']

# BASE_PATH = pathlib.Path('/media/UbuntuData3/Users_Data/amitp/PycharmProjects/mal-coral')

DATA_LOCATION = pathlib.Path('Matan/seg-data')
MODE = 'segmentation-rle'

## Universal configuration
DOWNLOAD_IMAGES = False  # Download data from labelbox. Set false for re-runs when data already exists locally
VALIDATION_RATIO = 0.15  #0.2 Validation data / training data ratio
NUM_CPU_THREADS = 8  # for multiprocess downloads
NUM_SAMPLE_LABELS = 0  # Use 0 to use all of the labeled training data from project. Otherwise specify number of labeled images to use. Use smaller number for faster iteration.
PRELABELING_THRESHOLD = 0.6 #0.6, 0.3  # minimum model inference confidence threshold to be uploaded to labelbox
HEADLESS_MODE = True  # Set True to skip previewing data or model results
SHOW_ANNOTATIONS = False

DETECTRON_DATASET_TRAINING_NAME = 'prelabeling-train'
DETECTRON_DATASET_VALIDATION_NAME = 'prelabeling-val'

EXISTING_JSON_TRAINING_PATH = 'traindataset_dict.pickle'
# EXISTING_JSON_TRAINING_PATH = None
EXISTING_JSON_VAL_PATH      = 'valdataset_dict.pickle'
# EXISTING_JSON_VAL_PATH      = None

#if you want to change network architecture
PROPOSAL_GENERATOR_NAME = "RPN" #"CoralRPN"
ROI_HEADS_NAME = "CoralStandardROIHeads" #for imbalbanced dataset #"StandardROIHeads"
BALANCED_WEIGHTS = None #bad programming, need to change

#in order to avoid annotating mouths
SKIP_MOUTHS = True

#to account for paratial annotated datset
TRAIN_WITH_MASK = True
MASK_TYPE = 'box'

IS_TRAINING = False
RESUME_TRAINING = False

# configurations for data utils that I took out
train = 'train'
val = 'val'
inference = 'inference'
masks = 'masks'
tmp = 'tmp'

# configuraions for training that I took out
EVAL_PERIOD = 500
DATALOADER_NUM_WORKERS = 0
IMS_PER_BATCH = 2
BASE_LR = 0.00125
MAX_ITER = 5000 #5000
BATCH_SIZE_PER_IMAGE = 256
CHECKPOINT_PERIOD = 500

DETECTIONS_PER_IMAGE = 300 #100
SIMPLIFY_VALUE_POLYGON = 2

ANCHOR_GENERATOR_SIZES = [[32], [64], [128], [256], [512]]
# ANCHOR_GENERATOR_SIZES = [[128], [256], [512]]
RPN_IN_FEATURES = ["p2", "p3", "p4", "p5", "p6"]
# RPN_IN_FEATURES = ["p4", "p5", "p6"]

INFERENCE_TO_AGISOFT = True # Save annotations to json in addition to uploading for LABELBOX
OUTPUT_PATH                 = 'outputs/singlecoliglooC2DipsFull.json' #name of the json file (save annotations)