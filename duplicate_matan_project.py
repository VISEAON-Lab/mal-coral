import labelbox
import labelbox as lb
from labelbox import LabelingFrontend
from labelbox.schema.media_type import MediaType
# from labelbox.data.annotation_types.data import ImageData
from config import *

client = lb.Client(LB_API_KEY, "https://api.labelbox.com/graphql")

# Create a new project
project = client.create_project(name="Amit_duplicated",
                                description="a duplication of mal-coral",
                                # media_type=ImageData)
                                media_type=MediaType.Image)

# Get the Labelbox editor (if using custom editor)
editor = next(client.get_labeling_frontends(where = LabelingFrontend.name == 'editor'))

# Get exsiting ontology (you can share an ontology across projects)
ontology = client.get_ontology("cl06bz333lkys0zbhctbv67e3")

# Get dataset
for dataset_id in DATASETS:
    dataset = client.get_dataset(dataset_id)
    # Attach the dataset to the project
    project.datasets.connect(dataset)
# dataset = client.get_datasets(DATASETS)

# Setup project with editor and normalized ontology
project.setup_editor(ontology) ## Default method
project.setup(editor, ontology.normalized) ## If using custom editor

# Upload labeling instructions
project.upsert_instructions("LOCAL_FILE_PATH (PDF or HTML)")



#Detach the dataset from the project
# project.datasets.disconnect(dataset)