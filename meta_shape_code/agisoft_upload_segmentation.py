import json
import numpy as np
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import connected_components
from functools import wraps
from time import time
from contextlib import contextmanager
from statistics import mean

SUFFIX         = '.JPG'
suffix_length  = len(SUFFIX)
GROUP_NAMES    = ['Single', 'Early Division', 'Mid Division', 'Late Division', 'Multi Division']
GROUP_COLORS   = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207)]
CONFIDENCE_THRESHOLD = 0.7 #The lowest condifence for which polygons will be uploaded
INTERSECTION_THRESHOLD  = 1.5 # the minimal threshold between the smaller shape and the intersection between the shapes
DIFF_THRESHOLD = 2.5e-4 #2.5e-4 #1e-4 #for projection error ration between 2d and 3d
PROJECTION_ERROR_NUM = 10
# UPLOAD_FROM    = 'network'
UPLOAD_FROM    = 'labelbox'
# NETWORK_JSON   = '/media/UbuntuData3/Users_Data/amitp/PycharmProjects/mal-coral/Matan/seg-data/outputs/predictions_for_agisoft_chunk3.json'
NETWORK_JSON   = '/media/UbuntuData3/Users_Data/amitp/PycharmProjects/mal-coral/Matan/seg-data/outputs/predictions_for_agisoft_after_annotations.json'
# NETWORK_JSON   = '/media/UbuntuData3/Users_Data/amitp/PycharmProjects/mal-coral/Matan/seg-data/outputs/predictions_for_agisoft_after_annotations_weight_chunk2_threshold_09.json'
# NETWORK_JSON   = '/media/UbuntuData3/Users_Data/amitp/PycharmProjects/mal-coral/Matan/seg-data/outputs/LobophylliaNrSmallColonies_PrincessLoboBig2_KazaSmallCOl.json'
# NETWORK_JSON   = '/media/UbuntuData3/Users_Data/amitp/PycharmProjects/mal-coral/Matan/seg-data/outputs/PrincessLoboBig1.json'
LABELBOX_JSON  = '/media/UbuntuData3/Users_Data/amitp/AgiSoft/export-2022-10-03.json'

class MyException(Exception):
    pass


# def transfer_mask_based_on_orientation(im_mask, orientation, stage):
#     if orientation == 1:
#         return im_mask
#     elif (stage == 'training' and orientation == 6) or (stage == 'inference' and orientation == 8):
#         return np.rot90(im_mask, 3)
#     elif (stage == 'training' and orientation == 8) or (stage == 'inference' and orientation == 6):
#         return np.rot90(im_mask)
#
# orientation = get_orientation(filename)
# transfer_mask_based_on_orientation(pred_mask.squeeze(), orientation, 'inference')
# filename = str(DATA_LOCATION/inference/datarow.uid) + extension

#calculate time for a code line
@contextmanager
def measuretime():
    start = time()
    try:
        yield
    finally:
        print(time() - start)

#calculate time for a function
def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r took: %2.4f sec' % (f.__name__, te-ts))
        return result
    return wrap

#the longest distance between two points in a single shape
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

def build_shape(camera_name, polygon, group, highest_confidence, total_confidence, points3d, bad_points, center_2d, center_3d):
    return {'img_name': camera_name, 'polygon': polygon, 'group': group, 'highest_confidence': highest_confidence,
            'total_confidence': total_confidence, 'points3d': points3d, 'bad_points': bad_points, 'center_2d': center_2d, 'center_3d': center_3d}

#the minimal distance between two shapes
def calc_minimal_dist_between_shapes(shapes1, shapes2):
    dist_3d = np.zeros((len(shapes1), len(shapes2)))
    shape1_arr = [np.array(shape['points3d']) for shape in shapes1]
    shape2_arr = [np.array(shape['points3d']) for shape in shapes2]
    for i, shape1 in enumerate(shape1_arr):
        for j, shape2 in enumerate(shape2_arr):
            dist_3d[i, j] = cdist(shape1, shape2).min()
    return dist_3d

#places 1 in i,j if shapes i and j intersection is bigger than a predefined threshold
def calculate_shapes_intersection(shapes1, shapes2, stage=None):
    intersection_mat = np.zeros((len(shapes2), len(shapes2))) #since we need NxN matrix for connected components, All the surplus rows are zero
    dist_3d = calc_minimal_dist_between_shapes(shapes1, shapes2)
    dist_thresholds = calc_dist_thresholds(shapes1)
    for idx1, (shape1, dist_threshold) in enumerate(zip(shapes1, dist_thresholds)):
        for idx2, shape2 in enumerate(shapes2):
            if (stage == 'between_grp_intersection' and shape1['group'] == shape2['group']) or (stage == 'in_grp_intersection' and shape1['group'] != shape2['group']) or dist_3d[idx1][idx2] > dist_threshold:
                continue
            if shape1['polygon'].intersects(shape2['polygon']):
                intersection_area = shape1['polygon'].intersection(shape2['polygon']).area
                if min(shape1['polygon'].area, shape2['polygon'].area) < intersection_area * INTERSECTION_THRESHOLD:
                    intersection_mat[idx1][idx2] = 1
    return intersection_mat

#delete indices from a list, expect the list to be sorted backwards
def adjust_list(l, indices_to_delete):
    for ind in indices_to_delete:
        del l[ind]
    return l

#create shape groups for the current chunk
def create_groups(group_names):
    for group_idx, group_name in enumerate(group_names):
        layer = chunk.shapes.addGroup()
        layer.label = group_name
        layer.color = GROUP_COLORS[group_idx]
    groups      = chunk.shapes.groups
    group_names = [group.label for group in groups]
    return group_names, groups


def create_metashape_polygons(groups, group_names, shapes):
    print('===============================================')
    print('==== begin stage create metashape polygons ====')
    print('===============================================')

    for shape in shapes:
        polygon                  = chunk.shapes.addShape()
        # polygon.label         = shape['group'] #str(shape['highest_confidence'])
        polygon.boundary_type    = Metashape.Shape.BoundaryType.OuterBoundary #don't know if necessary
        polygon.geometry.type    = Metashape.Geometry.Type.PolygonType
        polygon.geometry         = Metashape.Geometry.Polygon(shape['points3d'])
        polygon.group            = groups[group_names.index(shape['group'])]
        shape['agisoft_polygon'] = polygon
    print('finished creating metashape polygons')

def ratio_diff_2D_3D(arr2d, arr3d):
    successive_pnt_dif_2d = np.linalg.norm(arr2d[:-1] - arr2d[1:], axis=1)
    successive_pnt_dif_3d = np.linalg.norm(arr3d[:-1] - arr3d[1:], axis=1)
    ratio_diff = successive_pnt_dif_3d / successive_pnt_dif_2d
    adjusted_ratio_diff = np.abs(ratio_diff - np.median(ratio_diff))
    result = np.where(adjusted_ratio_diff > DIFF_THRESHOLD)
    return result[0]

# @timing
def filter_by_projection_errors(camera_name, points2d, points3d):
    arr2d = np.array(points2d)
    arr3d = np.array(points3d)
    ind_to_delete = ratio_diff_2D_3D(arr2d, arr3d)
    count = 0
    if np.size(ind_to_delete) == 0:
        return points2d, points3d, count

    while (np.size(ind_to_delete) > 0) and (count < PROJECTION_ERROR_NUM): #find a solution for the first point
        arr2d = np.delete(arr2d, ind_to_delete[0] + 1, 0)
        arr3d = np.delete(arr3d, ind_to_delete[0] + 1, 0)
        ind_to_delete = ratio_diff_2D_3D(arr2d, arr3d)
        count = count + 1
    # #temp begin
    # if camera_name == 'MTN_4900' and idx == 22:
    #     print('2d array: ', arr2d)
    #     print('3d array: ', arr3d)
    # #temp end
    # print('There are', len(points3d) - arr3d.shape[0], 'bad points in', camera_name, 'polygon number', idx)
    print('There are', len(points3d) - arr3d.shape[0], 'bad points in', camera_name)
    if count > PROJECTION_ERROR_NUM - 1:
        return None, None, None
    return arr2d.tolist(), [Metashape.Vector(elem) for elem in arr3d.tolist()], count

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

    def split_annotations_originated_from_camera(self, shapes_3d):
        camera_shapes = []
        indices_list = []

        for idx, shape in reversed(list(enumerate(shapes_3d))):
            if shape['img_name'] == self.camera_name:
                camera_shapes.append(shape)
                indices_list.append(idx)
        for idx in indices_list:
            del shapes_3d[idx]

        return shapes_3d, camera_shapes

    # @timing
    def unify_3D_overlapped_shapes(self, shapes_3d, current_camera_shapes):
        shapes_2d = current_camera_shapes + self.project_shapes_to_2d(shapes_3d) # To calculate also in image intersections
        indices_to_delete = np.zeros(len(shapes_2d))
        intersection_mat = calculate_shapes_intersection(current_camera_shapes, shapes_2d, 'in_grp_intersection')
        components, labels = connected_components(csgraph=intersection_mat, directed=True)
        for component in range(components):
            above_threshold_indices = np.where(labels == component)[0].tolist()
            if len(above_threshold_indices) > 1:
                confidence_list_aux = [shapes_2d[idx]['highest_confidence'] for idx in above_threshold_indices]
                best_polygon_index = above_threshold_indices[confidence_list_aux.index(max(confidence_list_aux))]
                above_threshold_indices.remove(best_polygon_index)
                for idx in above_threshold_indices:
                    shapes_2d[best_polygon_index]['total_confidence'] += shapes_2d[idx]['total_confidence']
                    indices_to_delete[idx] = 1
        return adjust_list(shapes_2d, np.where(indices_to_delete == 1)[0][::-1].tolist())

    def check_overlaps_between_groups(self, shapes_3d, camera_shapes):
        shapes_2d = self.project_shapes_to_2d(camera_shapes + shapes_3d)
        indices_to_delete = np.zeros(len(shapes_2d))
        confidence_list = [shape['total_confidence'] for shape in shapes_2d]
        intersection_mat = calculate_shapes_intersection(camera_shapes, shapes_2d, 'between_grp_intersection')
        components, labels = connected_components(csgraph=intersection_mat, directed=False)
        for component in range(components):
            above_threshold_indices = np.where(labels == component)[0].tolist()
            if len(above_threshold_indices) > 1:
                confidence_list_aux = [confidence_list[idx] for idx in above_threshold_indices]
                max_aux_idx = confidence_list_aux.index(max(confidence_list_aux))
                maximal_confidnece_group = shapes_2d[above_threshold_indices[max_aux_idx]]['group']
                for idx in above_threshold_indices:
                    if shapes_2d[idx]['group'] != maximal_confidnece_group:
                        indices_to_delete[idx] = 1
        return adjust_list(shapes_2d, np.where(indices_to_delete == 1)[0][::-1].tolist())

def find_selected_shape():
    for shape in chunk.shapes.shapes:
        if shape.selected:
            print(shape.key)

# @timing
def upload_shapes_from_network(camera, image_2D_shapes):
    current_camera_shapes = []
    for idx, polygon_2d in enumerate(image_2D_shapes['Objects']):
        if polygon_2d['confidence'] < CONFIDENCE_THRESHOLD:
            continue
        points2d = [list(pnt.values()) for pnt in polygon_2d['polygon']]
        points3d, bad_points = camera.project_points_to_3d(points2d)
        for idx in bad_points:
            del points2d[idx]
        points2d, points3d, bad_points = filter_by_projection_errors(camera.camera_name, points2d, points3d)
        if points2d is None:
            continue
        polygon = Polygon(points2d).buffer(0)
        # center_2d = list(map(mean, zip(*points2d)))
        center_2d = [polygon.centroid.x, polygon.centroid.y]
        center_3d, _ = camera.project_points_to_3d([center_2d])
        current_camera_shapes.append(build_shape(camera.camera_name, polygon, polygon_2d['group'], polygon_2d['confidence'],
                                                 polygon_2d['confidence'], points3d, bad_points, center_2d, center_3d))
    return current_camera_shapes

def upload_shapes_from_labelbox(camera, image_2D_shapes):
    current_camera_shapes = []
    for idx, polygon_2d in enumerate(image_2D_shapes['Label']['objects']):
        if polygon_2d['title'] != 'Segment':
            continue
        points2d = [list(pnt.values()) for pnt in polygon_2d['polygon']]
        points3d, bad_points = camera.project_points_to_3d(points2d)
        for idx in bad_points:
            del points2d[idx]
        points2d, points3d, bad_points = filter_by_projection_errors(camera.camera_name, points2d, points3d)
        if points2d is None:
            continue
        polygon = Polygon(points2d).buffer(0)
        # center_2d = list(map(mean, zip(*points2d)))
        center_2d = [polygon.centroid.x, polygon.centroid.y]
        center_3d, _ = camera.project_points_to_3d([center_2d])
        current_camera_shapes.append(build_shape(camera.camera_name, polygon,
                                                 polygon_2d['classifications'][0]['answer']['title'], 1, 1, points3d,
                                                 bad_points, center_2d, center_3d))
    return current_camera_shapes

def create_camera(cameras, camera_names, imgName, shapes_3d, idx):
    if imgName not in camera_names:
        return None
    print(f"=== the index of the image is {idx} ===")
    print('=== number of shapes before camera', imgName, ':', len(shapes_3d), '===')
    orientation = cameras[camera_names.index(imgName)].orientation
    print(f'The orientation of the camera is {orientation}')
    return CameraOperations(cameras[camera_names.index(imgName)])

# This function projects 2D shapes to 3D shapes and filter overlapping shapes from the same class
# Inputs:
# images_2D_shapes: list of images, where each image contains a list of 2D shapes
# cameras: list of cameras
# Outputs:
# shapes_3D: list of 3D shapes without in-group overlap
def create_3D_model_shapes(images_2D_shapes, cameras, camera_names):
    print('============================================')
    print('==== begin stage create 3D model shapes ====')
    print('============================================')
    shapes_3d = []

    #tmp
    # images_2D_shapes = [images_2D_shapes[119], images_2D_shapes[120]]

    for idx, image_2D_shapes in enumerate(images_2D_shapes):
        camera     = create_camera(cameras, camera_names, image_2D_shapes['External ID'][:-suffix_length], shapes_3d, idx)
        if camera is None:
            continue
        if camera.camera_center == None:
            print(f"camera_center of camera {camera.camera_name} is None")
            continue
        current_camera_shapes = project_shapes_to_3d(camera, image_2D_shapes)
        if len(current_camera_shapes) == 0:
            continue
        shapes_3d  = camera.unify_3D_overlapped_shapes(shapes_3d, current_camera_shapes)
    return shapes_3d

# @timing
def remove_duplications_from_model(images_2D_shapes, cameras, camera_names, shapes_3d):
    print('====================================================')
    print('==== begin stage remove duplications from model ====')
    print('====================================================')
    for idx, image_2D_shapes in enumerate(images_2D_shapes):
        camera    = create_camera(cameras, camera_names, image_2D_shapes['External ID'][:-suffix_length], shapes_3d, idx)
        if camera is None:
            continue
        if camera.camera_center == None:
            print(f"camera_center of camera {camera.camera_name} is None")
            continue
        shapes_3d, camera_shapes = camera.split_annotations_originated_from_camera(shapes_3d)
        if len(camera_shapes) == 0:
            continue
        shapes_3d = camera.check_overlaps_between_groups(shapes_3d, camera_shapes)
    return shapes_3d

if __name__ == '__main__':
    if  UPLOAD_FROM == 'network':
        project_shapes_to_3d = upload_shapes_from_network
        json_file     = NETWORK_JSON
    elif UPLOAD_FROM == 'labelbox':
        project_shapes_to_3d = upload_shapes_from_labelbox
        json_file     = LABELBOX_JSON
    else:
        raise Exception("upload_shapes from this file is not implemented")

    print('===========================================')
    print(f'======= upload shapes from {UPLOAD_FROM} =======')
    print('===========================================')

    chunk         = Metashape.app.document.chunk
    if not chunk.shapes:
        chunk.shapes     = Metashape.Shapes()
        chunk.shapes.crs = chunk.crs
    cameras       = chunk.cameras
    camera_names  = [camera.label for camera in cameras]
    [group_names, groups] = create_groups(GROUP_NAMES)

    with open(json_file, 'r') as f:
        images_2D_shapes = json.load(f)

    shapes_3d = create_3D_model_shapes(images_2D_shapes, cameras, camera_names)

    shapes_3d = remove_duplications_from_model(images_2D_shapes, cameras, camera_names, shapes_3d)

    create_metashape_polygons(groups, group_names, shapes_3d)

    # find_selected_shape()

    # shapes_3d_copy = shapes_3d.copy()
    # indices_to_delete = [0, 5, 17, 27, 167, 243]
    # indices_to_delete.sort(reverse=True)
    # shapes_3d = adjust_list(shapes_3d, indices_to_delete)

    # create_metashape_polygon(groups, group_names, shapes_3d)

    # dist_between_shapes = calc_dist_between_shapes(shapes_3d, shapes_3d, None)
    # dist_thresholds = np.array(calc_dist_thresholds(shapes_3d))
    # num_neighbours = (dist_between_shapes < dist_thresholds).sum(axis=1) - 1
    # shapes_3d = adjust_shapes(shapes_3d, np.where(num_neighbours == 0)[0][::-1].tolist())

    # faces_list = [list(face.tex_vertices) for face in chunk.model.faces]
    # vertices_list = [list(vertex.coord) for vertex in chunk.model.vertices]
    # mesh_vedo = vedo.Mesh([vertices_list, vertices_list])

    ###### new

    # shapes_3d_new = [{'points3d': shape.geometry.coordinates[0], 'area2d': 0} for shape in chunk.shapes]
    # shape_sizes = calc_dist_thresholds(shapes_3d_new)
    #
    # def project_shapes_to_2d_post_process(shapes, images_2D_shapes):
    #     for idx, image_2D_shapes in enumerate(images_2D_shapes):
    #         ##create camera
    #         camera = create_camera(cameras, camera_names, image_2D_shapes['External ID'][:-suffix_length], shapes_3d,
    #                                idx)
    #         if camera is None:
    #             continue
    #         if camera.camera_center == None:
    #             print(f"camera_center of camera {camera.camera_name} is None")
    #             continue
    #
    #         ##project all the shapes on all the images and find the image with the largest polygon size
    #         for shape_idx, shape in enumerate(shapes):
    #             points2d = [camera.camera.project(chunk.transform.matrix.inv().mulp(chunk.crs.unproject(pnt))) for pnt in shape['points3d']]
    #             points2d_new = [pnt for pnt in points2d if pnt is not None]
    #             if len(points2d) != len(points2d_new):
    #                 continue
    #             polygon = Polygon(points2d).buffer(0)
    #             area = polygon.area
    #             if shape['area2d'] < area: #check if this is the largest area for this polygon
    #                 center_2d = [polygon.centroid.x, polygon.centroid.y]
    #                 shapes_3d_new[shape_idx]['center_3d'], _ = camera.project_points_to_3d([center_2d])
    #
    #                 #check if this is the right range in terms of 3d distances
    #                 if np.linalg.norm(shapes_3d_new[shape_idx]['center_3d'][0] - shape['points3d'][0]) < shape_sizes[shape_idx]:
    #                     shape['area2d'] = area
    #                     shapes_3d_new[shape_idx]['camera_name'] = camera.camera.label
    #                     shapes_3d[shape_idx]['center_3d'] = shapes_3d_new[shape_idx]['center_3d']
    #
    #
    #     return shapes
    #
    # shapes_3d_new_2 = project_shapes_to_2d_post_process(shapes_3d_new, images_2D_shapes)

