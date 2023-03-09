import pathlib
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
import json
from scipy.spatial import ConvexHull

BASE_FOLDER = pathlib.Path('/media/UbuntuData3/Users_Data/amitp/PycharmProjects/mal-coral')
GRAPH_DATA_LOCATION = pathlib.Path('agisoft_extracted_data')
# PROJECT_NAME = 'PrincessLoboBig1'
# PROJECT_NAME = 'TrialforDataExtraction'
PROJECT_NAME = 'project_2'
(BASE_FOLDER / GRAPH_DATA_LOCATION / PROJECT_NAME).mkdir(exist_ok=True)
DATA_FILE_NAME = 'shapes_data_chunk3_labelbox_data_for_matan_cameras_from_file.csv'
filename = BASE_FOLDER / GRAPH_DATA_LOCATION / PROJECT_NAME / DATA_FILE_NAME
SUFFIX         = '.JPG'
suffix_length  = len(SUFFIX)
OVERLAP_FACTOR = 2


# UPLOAD_FROM    = 'network'
# UPLOAD_FROM    = 'labelbox'
UPLOAD_FROM    = 'agisoft'
# NETWORK_JSON   = '/media/UbuntuData3/Users_Data/amitp/PycharmProjects/mal-coral/Matan/seg-data/outputs/predictions_for_agisoft_chunk3.json'
NETWORK_JSON   = '/media/UbuntuData3/Users_Data/amitp/PycharmProjects/mal-coral/Matan/seg-data/outputs/predictions_for_agisoft_after_annotations.json'
# NETWORK_JSON   = '/media/UbuntuData3/Users_Data/amitp/PycharmProjects/mal-coral/Matan/seg-data/outputs/predictions_for_agisoft_after_annotations_weight_chunk2_threshold_09.json'
# NETWORK_JSON   = '/media/UbuntuData3/Users_Data/amitp/PycharmProjects/mal-coral/Matan/seg-data/outputs/LobophylliaNrSmallColonies_PrincessLoboBig2_KazaSmallCOl.json'
# NETWORK_JSON   = '/media/UbuntuData3/Users_Data/amitp/PycharmProjects/mal-coral/Matan/seg-data/outputs/Single_Col_8dataset.json'
# NETWORK_JSON   = '/media/UbuntuData3/Users_Data/amitp/PycharmProjects/mal-coral/Matan/seg-data/outputs/PrincessLoboBig1.json'
LABELBOX_JSON  = '/media/UbuntuData3/Users_Data/amitp/AgiSoft/export-2022-10-03.json'

######functions I tool from agisoft_upload_segmentation
class CameraOperations:
    def __init__(self, camera):
        self.camera        = camera
        self.camera_name   = camera.label
        self.camera_center = camera.center

    # @timing
    def project_shapes_to_2d(self, shapes):
        for shape_idx, shape in enumerate(shapes):
            points2d = [self.camera.project(chunk.transform.matrix.inv().mulp(chunk.crs.unproject(pnt))) for pnt in shape['points3d']]
            points2d = [pnt for pnt in points2d if pnt is not None]
            if len(points2d) > 2:
                polygon = Polygon(points2d).buffer(0)
                shapes[shape_idx]['polygon'] = polygon
            else:
                print(f"Shape in image {shape['img_name']} and center3D {shape['center_3d']} has only {len(points2d)} points when projecting to camera {self.camera_name}")
        return shapes

    # @timing
    def project_points_to_3d(self, points2d):
        points3d = []
        bad_points = []
        for idx, pnt in enumerate(points2d):
            point3d = chunk.model.pickPoint(self.camera_center, self.camera.unproject(pnt))
            if point3d is None:
                print(self.camera_name, ' has none 3d point')
                bad_points.append(idx)
                continue
            point3d = chunk.crs.project(chunk.transform.matrix.mulp(point3d))
            points3d.append(point3d)
        return points3d, bad_points

def create_camera(cameras, camera_names, imgName):
    if imgName not in camera_names:
        return None
    return CameraOperations(cameras[camera_names.index(imgName)])

def calc_dist_thresholds(shapes):
    dist_thresholds = []
    for shape in shapes:
        points = np.array(shape['points3d'])
        hull = ConvexHull(points)
        # Extract the points forming the hull
        hullpoints = points[hull.vertices, :]
        #finding the best pair
        hdist = cdist(hullpoints, hullpoints, metric='euclidean')
        # Get the farthest apart points
        bestpair = np.unravel_index(hdist.argmax(), hdist.shape)
        dist_thresholds.append(np.linalg.norm([hullpoints[bestpair[0]] - hullpoints[bestpair[1]]]))
    return dist_thresholds

########new function
def project_shapes_to_2d_post_process(images_2D_shapes, shapes, shape_sizes):
    for image_2D_shapes in images_2D_shapes:
        ##create camera
        camera = create_camera(cameras, camera_names, image_2D_shapes['External ID'][:-suffix_length])
        if camera is None:
            continue
        if camera.camera_center == None:
            print(f"camera_center of camera {camera.camera_name} is None")
            continue

        ##project all the shapes on all the images and find the image with the largest polygon size
        for shape_idx, shape in enumerate(shapes):
            points2d = [camera.camera.project(chunk.transform.matrix.inv().mulp(chunk.crs.unproject(pnt))) for pnt in shape['points3d']]
            points2d_new = [pnt for pnt in points2d if pnt is not None]
            if len(points2d) != len(points2d_new):
                continue
            polygon = Polygon(points2d).buffer(0)
            area = polygon.area
            if shape['area2d'] < area: #check if this is the largest area for this polygon
                center_2d = [polygon.centroid.x, polygon.centroid.y]
                center_3d, _ = camera.project_points_to_3d([center_2d])
                #new
                if len(center_3d) == 0:
                    print(f"problem in center3d in camera {camera.camera_name}")
                    continue
                center_3d = list(center_3d[0])
                if len(center_3d) != 3:
                    print(f"problem in center3d in camera {camera.camera_name}")
                    continue

                #check if this is the right range in terms of 3d distances
                if np.linalg.norm(center_3d - shape['points3d'][0]) < shape_sizes[shape_idx]:
                    shapes[shape_idx]['area2d'] = area
                    shapes[shape_idx]['camera_name'] = camera.camera.label
                    # shapes[shape_idx]['center_3d'] = list(center_3d[0])
                    shapes[shape_idx]['center_3d'] = center_3d #new


    return shapes

def check_enabled_cameras(cameras):
    enabled_list = [camera for camera in cameras if camera.enabled]
    print(f"*****images to label: {len(enabled_list)}****")
    for camera in enabled_list:
        print(camera.label + " is enabled.")
    return enabled_list

if __name__ == '__main__':
    chunk         = Metashape.app.document.chunk
    cameras       = chunk.cameras
    camera_names  = [camera.label for camera in cameras]

    if  UPLOAD_FROM == 'network':
        json_file     = NETWORK_JSON
        with open(json_file, 'r') as f:
            images_2D_shapes = json.load(f)
    elif UPLOAD_FROM == 'labelbox':
        json_file     = LABELBOX_JSON
        with open(json_file, 'r') as f:
            images_2D_shapes = json.load(f)
    elif UPLOAD_FROM == 'agisoft':
        chunk.reduceOverlap(OVERLAP_FACTOR)
        images_2D_shapes = [{'External ID': f"{camera.label}{SUFFIX}"} for camera in cameras if camera.enabled]
    else:
        raise Exception("upload_shapes from this file is not implemented")

    shapes_data = {}

    shapes_data['perimeter3D']        = [shape.perimeter3D() for shape in chunk.shapes]
    shapes_data['group']              = [shape.group.label for shape in chunk.shapes]

    shapes_data['points3d']           = []
    shapes = []
    for shape in chunk.shapes:
        points3d = np.array([np.array(pnt3d) for pnt3d in shape.geometry.coordinates[0]])
        shapes_data['points3d'].append(points3d)
        shapes.append({'points3d': points3d, 'area2d': 0})

    shape_sizes = calc_dist_thresholds(shapes)

    shapes = project_shapes_to_2d_post_process(images_2D_shapes, shapes, shape_sizes)

    shapes_data['center3D']        = [shape['center_3d'] for shape in shapes]
    shapes_data['img_name']        = [shape['camera_name'] for shape in shapes]

    df = pd.DataFrame(data=shapes_data)
    df.to_csv(filename, index=False)

