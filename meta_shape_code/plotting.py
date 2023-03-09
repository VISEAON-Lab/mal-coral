import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pathlib
import pyvista as pv
import re
import pymeshlab
import torch
from dgl.geometry import farthest_point_sampler

import open3d as o3d
import potpourri3d as pp3d
import vedo
from sklearn.neighbors import KDTree
from create_masks import get_face_areas
from create_masks import cut_path_segment


# extract points from vertices
def extract_points_from_text(text):
    string_numbers = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", text)
    points_array = np.array(string_numbers, dtype=np.float32).reshape(-1, 3)
    return points_array

def create_poly_data(vertices, faces=None, colors=None):
    if faces is not None:
        stack_vec = faces.shape[1] * np.ones((faces.shape[0], 1), dtype=int)
        pv_object = pv.PolyData(vertices, np.hstack((stack_vec, faces)))

    else:
        pv_object = pv.PolyData(vertices)

    if colors is not None:
        pv_object['colors'] = colors / 256

    return pv_object

def find_vertices_in_geodesic_shape(vertices, dist, max_geodesic_dist, extra_space=0):
    max_geodesic_dist = max_geodesic_dist + extra_space * max_geodesic_dist
    vetrtices_indices = np.array((dist < max_geodesic_dist).nonzero()).squeeze()
    return vetrtices_indices

def find_vertices_in_rectangle(vertices, shapes_centers, extra_space=0):
    #find euclidean extreme points
    min_shapes_centers = shapes_centers.min(axis=0)
    max_shapes_centers = shapes_centers.max(axis=0)
    size_rect = np.absolute(max_shapes_centers - min_shapes_centers)
    min_shapes_centers = min_shapes_centers - extra_space * size_rect
    max_shapes_centers = max_shapes_centers + extra_space * size_rect
    vetrtices_indices = np.array((((vertices > min_shapes_centers) & (vertices < max_shapes_centers)).sum(axis=1) == 3).nonzero()).squeeze()
    return vetrtices_indices


def farthest_points_sample(vertices, n_samples, start_idx=0, masked_vertices_indices=None, masked_vertices=None, faces=None, colors=None, visualize=False):
    """

    :param vertices: vertices of the mesh
    :param faces: faces of the mesh
    :param n_samples: number of sampled points
    :param start_idx: the index of the starting sampling
    :param colors: color per vertex
    :return: the indices of the sampled vertices anf the vertices of the patch
    """

    if masked_vertices is None:
        masked_vertices = vertices

    fps_idx = torch.squeeze(
        farthest_point_sampler(torch.unsqueeze(torch.Tensor(masked_vertices), 0), n_samples, start_idx=start_idx)).numpy()

    if masked_vertices_indices is not None:
        fps_idx = masked_vertices_indices[fps_idx]

    if visualize:
        plotter = pv.Plotter()

        if colors is None:
            plotter.add_mesh(create_poly_data(vertices, faces), scalars=vertices[:, 2], colormap='rainbow')
        else:
            plotter.add_mesh(create_poly_data(vertices, faces, colors=colors), show_edges=False, scalars='colors',
                             rgb=True, preference='point')
        plotter.add_mesh(create_poly_data(vertices[fps_idx]), style='points', color='red', point_size=15.,
                         render_points_as_spheres=True, show_scalar_bar=False)
        plotter.show()

    return fps_idx, vertices[fps_idx]

def read_plyfile(file_path, cls='meshlab'):
    if cls == 'meshlab':

        # read 3d model - pymeshlab
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(file_path))
        ms.compute_color_from_texture_per_vertex()
        meshlab = ms.current_mesh()

        vertices = np.asarray(meshlab.vertex_matrix())
        faces = np.asarray(meshlab.face_matrix())
        if meshlab.has_vertex_color():
            colors = meshlab.vertex_color_matrix()[:, 0:3]
        else:
            colors = None

    if cls == 'o3d':
        plydata = o3d.io.read_triangle_mesh(filename=str(file_path))
        vertices = np.asarray(plydata.vertices)
        faces = np.asarray(plydata.triangles)
        try:
            colors = 255 * np.asarray(plydata.vertex_colors)
        except:
            colors = None

    elif cls == 'pp3d':
        vertices, faces = pp3d.read_mesh(str(file_path))
        colors = None

    else:
        ValueError(f"'cls' argument should be one of: 'o3d' or 'pp3d'")

    return vertices, faces, colors


def build_pyvista_mesh(vertices, faces=None, colors=None):
    # creating pyvista object

    if faces is not None:

        stack_vec = faces.shape[1] * np.ones((faces.shape[0], 1), dtype=int)
        mesh_pv = pv.PolyData(vertices, np.hstack((stack_vec, faces)))

    else:
        mesh_pv = pv.PolyData(vertices)

    if colors is not None:
        mesh_pv['colors'] = colors

    return mesh_pv


def plot_histogram(bins, data, frame, xlabel, ylabel, title, space=0.985):
    fig = plt.figure(tight_layout=True)
    plt.hist(data[frame], bins=bins, rwidth=0.8)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    fig.suptitle(title, fontsize=16)
    plt.show()


def plot_histogram_per_grp(bins, data, frame, xlabel, ylabel, title, grp, space=1):
    grouped = data.groupby(grp)
    data_list = []
    labels_list = []
    for group in GROUP_NAMES:
        for grp in grouped:
            if grp[0] == group:
                labels_list.append(group)
                data_list.append(grp[1][frame])
    fig, axs = plt.subplots(len(data_list), 1, figsize=(7, 11), tight_layout=True)
    fig.suptitle(title, fontsize=18)
    for idx, elem_grp in enumerate(labels_list):
        axs[idx].hist(data_list[idx], bins=bins, rwidth=0.8, color=GROUP_COLORS2[GROUP_NAMES.index(elem_grp)])
        axs[idx].set_ylabel(ylabel=ylabel, fontsize=16)
        if idx == len(data_list) - 1:
            axs[idx].set_xlabel(xlabel=xlabel, fontsize=16)
        axs[idx].legend(loc='upper left', labels=[elem_grp])
    plt.show()


def plot_scatter(data, frame1, frame2, xlabel, ylabel, title):
    grouped = data.groupby('group')
    data_list1 = [grp[1][frame1] for grp in grouped]
    data_list2 = [grp[1][frame2] for grp in grouped]
    labels_list = [grp[0] for grp in grouped]
    fig = plt.figure(tight_layout=True)
    for idx, elem_grp in enumerate(labels_list):
        plt.scatter(data_list1[idx], data_list2[idx], color=GROUP_COLORS2[GROUP_NAMES.index(elem_grp)], label=elem_grp)
    plt.legend()
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    fig.suptitle(title, fontsize=16)
    plt.show()

def create_fewer_groups(y, coral_area_ratio, previous_groups, new_group):

    coral_area_ratio[new_group] = 0
    for grp in previous_groups:
        coral_area_ratio[new_group] = coral_area_ratio[new_group] + coral_area_ratio[grp]

    for i, y_i in enumerate(y):
        tmp = 0
        for grp in previous_groups:
            tmp = tmp + y_i[grp]
        y[i][new_group] = tmp
    return y, coral_area_ratio

def plot_graph(data, x, y, xlabel, ylabel, title, coral_area_ratio=None, fewer=False):
    if coral_area_ratio is None:
        coral_area_ratio = {grp: 1 / len(GROUP_NAMES) for grp in GROUP_NAMES}

    y = data[y]
    x = [data[x][i] for i, y_i in enumerate(y) if y_i is not None]
    y = [y_i for i, y_i in enumerate(y) if y_i is not None]

    group_names = GROUP_NAMES

    if fewer:
        previous_groups = [item for item in GROUP_NAMES if item not in GROUP_NAMES_FEWER]
        new_group = [item for item in GROUP_NAMES_FEWER if item not in GROUP_NAMES]
        y, coral_area_ratio = create_fewer_groups(y, coral_area_ratio, previous_groups, new_group[0])
        group_names = GROUP_NAMES_FEWER

    fig = plt.figure(tight_layout=True)
    for idx, elem_grp in enumerate(group_names):
        if coral_area_ratio[elem_grp] != 0:
            plt.scatter(x, [y_ii[elem_grp] / coral_area_ratio[elem_grp] for y_ii in y], color=GROUP_COLORS2[group_names.index(elem_grp)], label=elem_grp)
    plt.legend()
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    fig.suptitle(title, fontsize=16)
    plt.show()

def enter_paths_to_shapes_data(shapes_data, paths):
    fields = {key: [] for key in paths[0].keys() if key !='group'}
    for path in paths:
        for field in fields.keys():
            fields[field].append(path[field])
    for key in fields.keys():
        shapes_data[key] = fields[key]
    return shapes_data




def define_projects():
    projects_dict = {}
    projects_dict['project2_chunk1'] = {"project_name": 'project_2',
                                        "data_file_name": 'shapes_data_chunk1_labelbox_data_for_matan.csv',
                                        "agisoft_dataset_filename": pathlib.Path('project2_chunk1_cc_me.ply'),
                                        "manual_geodesic_center": [-0.35609185695648193, 0.5548081398010254, -3.6885228157043457]
                                        }

    projects_dict['project2_chunk3'] = {"project_name": 'project_2',
                                        "data_file_name": 'shapes_data_chunk3_labelbox_data_for_matan.csv',
                                        # "data_file_name": 'shapes_data_chunk3_labelbox_data_for_matan_cameras_from_file.csv',
                                        "agisoft_dataset_filename": pathlib.Path('project2_chunk3_cc_me.ply'),
                                        "manual_geodesic_center": [-0.7222669124603271, 0.6427904963493347, -6.467058181762695]
                                        }

    projects_dict['PrincessLoboBig1'] = {"project_name": 'PrincessLoboBig1',
                                        "data_file_name": 'shapes_data_PrincessLoboBig1.csv'
                                        # "agisoft_dataset_filename": pathlib.Path('project2_chunk3_cc_me.ply'),
                                        # "manual_geodesic_center": [-0.7222669124603271, 0.6427904963493347,
                                        #                            -6.467058181762695]
                                        }

    return projects_dict



if __name__ == '__main__':

    PROJECT_FULL_NAME = 'project2_chunk3'
    print(f"========== Defining project for {PROJECT_FULL_NAME} ==========")
    GROUP_NAMES = ['Single', 'Early Division', 'Mid Division', 'Late Division', 'Multi Division']
    GROUP_NAMES_FEWER = ['Single', 'Double Division', 'Multi Division']
    GROUP_COLORS = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189), (140, 86, 75),
                    (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207)]
    GROUP_COLORS2 = np.array(GROUP_COLORS, dtype=int).reshape((len(GROUP_COLORS), 3)) / 255
    GROUP_COLORS_FEWER = GROUP_COLORS2
    GROUP_COLORS2_dict = {GRP: GROUP_COLORS2[idx] for idx, GRP in enumerate(GROUP_NAMES)}
    GROUP_COLORS2_FEWER_dict = {GRP: GROUP_COLORS2[idx] for idx, GRP in enumerate(GROUP_NAMES_FEWER)}

    projects_dict = define_projects()

    GRAPH_DATA_LOCATION = pathlib.Path('../agisoft_extracted_data')
    agisoft_dataset_dir = pathlib.Path('../agisoft_extracted_data', projects_dict[PROJECT_FULL_NAME]["project_name"], 'extracted_model_data')

    N_SAMPLES = 100
    R = 0.1
    VISUALIZATION = False

    filename = GRAPH_DATA_LOCATION / projects_dict[PROJECT_FULL_NAME]["project_name"] / projects_dict[PROJECT_FULL_NAME]["data_file_name"]

    agisoft_dataset_path = agisoft_dataset_dir / projects_dict[PROJECT_FULL_NAME]["agisoft_dataset_filename"]


    print("========== Extracting the csv file ==========")
    shapes_data = pd.read_csv(filename)
    num_of_shapes = shapes_data.shape[0]
    shapes_data['points3d'] = shapes_data['points3d'].apply(extract_points_from_text)
    shapes_data['center3D'] = shapes_data['center3D'].apply(extract_points_from_text)
    shapes_centers = np.array(shapes_data['center3D'].values.tolist()).astype(float).squeeze()

    euclidian_center_of_shapes = shapes_centers.mean(axis=0)
    shapes_data['center_euclidean_dist'] = np.linalg.norm(shapes_centers - euclidian_center_of_shapes, axis=1).tolist()

    print(f"========== Number of shapes in the coral is {num_of_shapes}  ==========")

    num_bins = min(int(np.ceil(num_of_shapes / 5)), 30)
    print(f"========== Defining number of bins in the graph as {num_bins}, the number of sampled points as {N_SAMPLES} and the radius is {R} ==========")

    print("========== Building vedo, pyvista and solver mesh and finding geodesic vertices for shapes ==========")
    vertices, faces, colors = read_plyfile(file_path=agisoft_dataset_path)
    mesh_vedo = vedo.Mesh([vertices, faces])
    mesh_pv = build_pyvista_mesh(vertices, faces, colors)
    solver = pp3d.MeshHeatMethodDistanceSolver(vertices, faces)

    print("========== Calculating geodesic distances of shapes from geodesic center ==========")
    projects_dict[PROJECT_FULL_NAME]["manual_geodesic_center_ind"] = mesh_vedo.closestPoint(projects_dict[PROJECT_FULL_NAME]["manual_geodesic_center"], returnPointId=True)
    dist = solver.compute_distance(projects_dict[PROJECT_FULL_NAME]["manual_geodesic_center_ind"])

    shapes_data["center3D_geodesic"] = [mesh_vedo.closestPoint(center, returnPointId=True) for center in shapes_centers]

    points3d_geodesic = []
    for shape_points3d in shapes_data['points3d']:
        points3d_geodesic.append([mesh_vedo.closestPoint(pnt, returnPointId=True) for pnt in shape_points3d])
    shapes_data["points3d_geodesic"] = points3d_geodesic

    center_distances_geodesic = dist[shapes_data["center3D_geodesic"]]
    shapes_data['center_distances_geodesic'] = center_distances_geodesic.tolist()

    masked_vertices_indices_geodesic = find_vertices_in_geodesic_shape(vertices, dist, max(center_distances_geodesic), extra_space=0)
    masked_vertices_indices_rectanbgle = find_vertices_in_rectangle(vertices, shapes_centers, extra_space=0)
    masked_vertices_indices = np.intersect1d(masked_vertices_indices_geodesic, masked_vertices_indices_rectanbgle)
    masked_vertices = vertices[masked_vertices_indices]

    sampled_points = {}
    sampled_points['idx'], _ = farthest_points_sample(vertices, masked_vertices=masked_vertices, masked_vertices_indices=masked_vertices_indices, faces=faces, n_samples=N_SAMPLES, start_idx=0, visualize=False)
    sampled_points['geodesic_dist'] = dist[sampled_points['idx']]

    print("=======================================================================================")
    print("=============================== Preforming segmentation ===============================")
    print("=======================================================================================")

    paths = cut_path_segment(vertices, faces, polygons_points_ind=shapes_data['points3d_geodesic'], centrals_ind=shapes_data["center3D_geodesic"], group=shapes_data['group'])

    vertices_ind = np.full((vertices.shape[0],), len(GROUP_NAMES), dtype=int)
    faces_ind = np.full((faces.shape[0],), len(GROUP_NAMES), dtype=int)
    faces_area = get_face_areas(vertices, faces)
    for path in paths:
        vertices_ind[path['segment_vertex_ind']] = GROUP_NAMES.index(path['group'])
        faces_ind[path['segment_face_idx']] = GROUP_NAMES.index(path['group'])
        path['segment_face_area'] = np.sum(faces_area[path['segment_face_idx']])

    shapes_data = enter_paths_to_shapes_data(shapes_data, paths)

    #### visualization
    if VISUALIZATION:
        pv.set_plot_theme('document')
        a = pv.Plotter(shape=(1, 2), border=False)

        a.subplot(0, 0)
        a.add_text("Original mesh", font_size=24)
        a.add_mesh(mesh_pv, show_edges=False, scalars='colors', rgb=True, preference='point')
        a.subplot(0, 1)
        a.add_text("Segmented version", font_size=24)
        a.add_mesh(mesh_pv, show_edges=False, scalars='colors', rgb=True, preference='point')
        for i, grp in enumerate(GROUP_NAMES):
            grp_vertices = vertices_ind == i
            face_vertices = faces_ind == i
            if vertices[grp_vertices].any():
                # a.add_mesh(build_pyvista_mesh(vertices[grp_vertices]), color=GROUP_COLORS2[i], render_points_as_spheres=True,
                #            point_size=12.)
                tmp_vertices = np.cumsum(grp_vertices) - 1
                tmp_faces = np.array([tmp_vertices[face] for face in faces[face_vertices]])
                a.add_mesh(build_pyvista_mesh(vertices[grp_vertices], faces=tmp_faces),
                           color=GROUP_COLORS2[i],
                           render_points_as_spheres=True,
                           point_size=12.)
        a.link_views()  # link all the views

        a.show()

    print("=======================================================================================")
    print("==================== Calculating ratio of groups per sampled point ====================")
    print("=======================================================================================")
    sampled_pnt_pv = vertices[sampled_points['idx']]
    tree = KDTree(vertices, leaf_size=vertices.shape[0] + 1)

    ind2 = tree.query_radius(sampled_pnt_pv, r=R) # indices of neighbors within distance r
    vertices_per_point = [list(ind2_ii) for ind2_ii in ind2]

    #new
    faces_per_point = []
    area_faces_per_point = []
    for ind2_ii in ind2:
        tmp = np.where(np.isin(faces, ind2_ii).sum(axis=1) == 3)[0]
        faces_per_point.append(faces_ind[tmp])
        area_faces_per_point.append(faces_area[tmp])

    #for polyp centers
    shapes_centers_pnt_pv = vertices[shapes_data["center3D_geodesic"]]
    vertices_shapes_centers_per_point_aux = tree.query_radius(shapes_centers_pnt_pv, r=R)
    vertices_shapes_centers_per_point = [list(ind2_ii) for ind2_ii in vertices_shapes_centers_per_point_aux]


    #### visualization
    if VISUALIZATION:
        p = pv.Plotter()
        p.add_mesh(mesh_pv)

        # visualise single point
        top = mesh_pv.extract_points(ind2[0])
        random_color = np.random.random(3)
        p.add_mesh(top, color=random_color)

        p.show()

    coral_area_ratio = {grp: faces_area[faces_ind == i].sum() for i, grp in enumerate(GROUP_NAMES)}
    coral_area_ratio = {grp: val / sum(coral_area_ratio.values()) for grp, val in coral_area_ratio.items()} #normalizing

    sampled_points["ratios_list"] = []
    for faces_per_point_ii, area_faces_per_point_ii in zip(faces_per_point, area_faces_per_point):
        point_ratios = {grp: area_faces_per_point_ii[faces_per_point_ii == i].sum() for i, grp in enumerate(GROUP_NAMES)}
        if sum(point_ratios.values()) != 0: #To avoid points without segments
            sampled_points["ratios_list"].append({key: value / sum(point_ratios.values()) for key, value in point_ratios.items()})
        else:
            sampled_points["ratios_list"].append(None)

    print("==================== Calculating curvature ====================")
    curv_mean = mesh_pv.curvature(curv_type='mean')
    curv_gaussian = mesh_pv.curvature(curv_type='gaussian')
    curv_maximum = mesh_pv.curvature(curv_type='maximum')
    curv_minimum = mesh_pv.curvature(curv_type='minimum')
    curv = np.sqrt(np.square(curv_maximum) + np.square(curv_minimum))

    # clipping_threshold = np.percentile(curv, 95)
    # curv_clipped = np.clip(curv, 0, clipping_threshold)
    # mesh_pv.plot(scalars=curv_clipped)

    #calculating curvature of every area around sampled point
    #for every point above the 95 percentile in its surrounding puts the value of the median
    sampled_points['curvature'] = []
    for vertices_per_point_ii in vertices_per_point:
        curv_point_ii = curv[vertices_per_point_ii]
        curv_point_ii_med = np.percentile(curv_point_ii, 50)
        curv_point_ii[curv_point_ii > np.percentile(curv_point_ii, 95)] = curv_point_ii_med
        sampled_points['curvature'].append(curv_point_ii.mean())

    if VISUALIZATION:
        vertices_with_curv = np.unique(np.concatenate(vertices_per_point)) #indices of vertices in the sampled points
        vertices_with_curv_count = np.zeros_like(vertices_with_curv) #their number of apperances in the sampled points
        vertices_with_curv_val = np.zeros_like(vertices_with_curv) #the values

        #calculates their curvature across the sampled points
        for ii, vertices_per_point_ii in enumerate(vertices_per_point):
            indices = np.searchsorted(vertices_with_curv, vertices_per_point_ii)
            vertices_with_curv_count[indices] = vertices_with_curv_count[indices] + 1
            vertices_with_curv_val[indices] = vertices_with_curv_val[indices] + sampled_points['curvature'][ii]
        vertices_with_curv_mean = vertices_with_curv_val / vertices_with_curv_count

        #put all other vertices with curvature zero and all the calculated vertices with their mean curvature
        curv_points_of_interest = np.zeros(curv.shape[0])
        curv_points_of_interest[vertices_with_curv] = vertices_with_curv_mean
        mesh_pv.plot(scalars=curv_points_of_interest)

    shapes_data["center_curvature"] = curv[shapes_data["center3D_geodesic"]]
    center_area_curvature = []
    for vertices_shapes_centers_per_point_ii in vertices_shapes_centers_per_point:
        center_area_curvature.append(curv[vertices_shapes_centers_per_point_ii].mean())
    shapes_data["center_area_curvature"] = center_area_curvature

    print("======================================================================================")
    print("=================================== Begin plotting ====================================")
    print("======================================================================================")

    # Graph for area ratio per curvature
    print("========== Plotting the graphs for area ratio per curvature ==========")
    plot_graph(sampled_points, 'curvature', 'ratios_list', 'Curvature', 'Ratios of groups',
               'Ratios of groups according to the curvature', coral_area_ratio, fewer=True)

    # Graph for number of polyps per perimeter
    print("========== Plotting the graphs for number of polyps per perimeter ==========")
    bins = np.linspace(min(shapes_data['perimeter3D']), max(shapes_data['perimeter3D']), num_bins)
    plot_histogram(bins, shapes_data, 'perimeter3D', 'Perimeter3D', 'Quantity', 'Number of polyps per perimeter')
    plot_histogram_per_grp(bins, shapes_data, 'perimeter3D', 'Perimeter3D', 'Quantity',
                           'Number of polyps per perimeter per group', grp='group')

    # Graph for number of polyps per area
    print("========== Plotting the graphs for number of polyps per area ==========")
    bins = np.linspace(min(shapes_data['segment_face_area']), max(shapes_data['segment_face_area']), num_bins)
    plot_histogram(bins, shapes_data, 'segment_face_area', 'segment face area', 'Quantity', 'Number of polyps per area')
    plot_histogram_per_grp(bins, shapes_data, 'segment_face_area', 'segment face area', 'Quantity',
                           'Number of polyps per area per group', grp='group')

    # Graph for number of polyps according to the center
    print("========== Plotting the number of polyps per euclidean distance from center ==========")
    bins = np.linspace(min(shapes_data['center_euclidean_dist']), max(shapes_data['center_euclidean_dist']), num_bins)
    plot_histogram(bins, shapes_data, 'center_euclidean_dist', 'Euclidean distance from the center', 'Quantity',
                   'Number of polyps according to their euclidean distance \n from the center')
    plot_histogram_per_grp(bins, shapes_data, 'center_euclidean_dist', 'Euclidean distance from the center', 'Quantity',
                           'Number of polyps according to their euclidean distance \n from the center per group', grp='group')
    plot_scatter(shapes_data, 'perimeter3D', 'center_euclidean_dist', 'Perimeter3D', 'Distance from the center',
                 'Perimeter according to the distance from center')

    # Graph for number of polyps according to geodesic distance from center
    print("========== Plotting the number of polyps per geodesic distance from center ==========")
    bins = np.linspace(min(shapes_data['center_distances_geodesic']), max(shapes_data['center_distances_geodesic']),
                       num_bins)
    plot_histogram(bins, shapes_data, 'center_distances_geodesic', 'Geodesic distance from center', 'Quantity',
                   'Number of polyps according to their geodesic distance \n from the center')
    plot_histogram_per_grp(bins, shapes_data, 'center_distances_geodesic', 'Geodesic distance from center', 'Quantity',
                           'Number of polyps according to their geodesic distance \n from the center per group',
                           grp='group')
    plot_scatter(shapes_data, 'perimeter3D', 'center_distances_geodesic', 'Perimeter3D',
                 'Geodesic distance from the center', 'Perimeter according to the \n geodesic distance from center')

    print("========== Plotting the number of polyps according to center curvature ==========")
    bins = np.linspace(min(shapes_data['center_curvature']), max(shapes_data['center_curvature']), num_bins)
    plot_histogram(bins, shapes_data, 'center_curvature', 'Center curvature', 'Quantity',
                   'Number of polyps according to their center curvature')
    plot_histogram_per_grp(bins, shapes_data, 'center_curvature', 'Center curvature', 'Quantity',
                           'Number of polyps according to their center curvature \n per group', grp='group')
    plot_scatter(shapes_data, 'perimeter3D', 'center_curvature', 'Perimeter3D', 'Center curvature',
                 'Perimeter according to the Center curvature')

    print("========== Plotting the number of polyps according to center area curvature ==========")
    bins = np.linspace(min(shapes_data['center_area_curvature']), max(shapes_data['center_area_curvature']), num_bins)
    plot_histogram(bins, shapes_data, 'center_area_curvature', 'Center area curvature', 'Quantity',
                   'Number of polyps according to their area curvature')
    plot_histogram_per_grp(bins, shapes_data, 'center_area_curvature', 'Center area curvature', 'Quantity',
                           'Number of polyps according to their area curvature \n per group', grp='group')
    plot_scatter(shapes_data, 'perimeter3D', 'center_area_curvature', 'Perimeter3D', 'Center area curvature',
                 'Perimeter according to the Center area curvature')

    print("========== Finished ==========")
