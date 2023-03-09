import pathlib
import pandas as pd
from statistics import mean
import numpy as np
# import pickle
BASE_FOLDER = pathlib.Path('/media/UbuntuData3/Users_Data/amitp/PycharmProjects/mal-coral')
GRAPH_DATA_LOCATION = pathlib.Path('agisoft_extracted_data')
# PROJECT_NAME = 'PrincessLoboBig1'
PROJECT_NAME = 'project_2'
(BASE_FOLDER / GRAPH_DATA_LOCATION / PROJECT_NAME).mkdir(exist_ok=True)
DATA_FILE_NAME = 'shapes_data_chunk1_labelbox_tmp.csv'
filename = BASE_FOLDER / GRAPH_DATA_LOCATION / PROJECT_NAME / DATA_FILE_NAME

shapes_data = {'img_name': [], 'perimeter3D': [], 'group': [], 'center3D': [], 'total_confidence': [], 'highest_confidence': [], 'bad_points': []}

shapes_data['img_name']           = [shape['img_name'] for shape in shapes_3d]
shapes_data['perimeter3D']        = [shape['agisoft_polygon'].perimeter3D() for shape in shapes_3d]
# shapes_data['group']              = [shape['agisoft_polygon'].group.label for shape in shapes_3d]
shapes_data['group']              = [shape['group'] for shape in shapes_3d]
# shapes_data['center3D']           = [list(map(mean, zip(*shape['agisoft_polygon'].geometry.coordinates[0]))) for shape in shapes_3d]
# shapes_data['center3D']           = [list(map(mean, zip(*shape['points3d']))) for shape in shapes_3d]
shapes_data['center3D']           = [shape['center_3d'] for shape in shapes_3d]
shapes_data['img_name']           = [shape['img_name'] for shape in shapes_3d]
shapes_data['total_confidence']   = [shape['total_confidence'] for shape in shapes_3d]
shapes_data['highest_confidence'] = [shape['highest_confidence'] for shape in shapes_3d]
shapes_data['bad_points']         = [shape['bad_points'] for shape in shapes_3d]

shapes_data['points3d']           = []
for shape in shapes_3d:
    shapes_data['points3d'].append(np.array([np.array(pnt3d) for pnt3d in shape['points3d']]))


df = pd.DataFrame(data=shapes_data)
df.to_csv(filename, index=False)
# with open(BASE_FOLDER / GRAPH_DATA_LOCATION / DATA_FILE_NAME, "wb") as fp:
#     pickle.dump(shapes_data, fp)



