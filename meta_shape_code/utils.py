import os
import pyvista as pv
import numpy as np
from numpy import linalg as LA
# import matplotlib
import scipy
from plyfile import PlyData, PlyElement
import open3d as o3d
from os.path import join as pjoin
import gdist
import itertools
import networkx as nx
import scipy.sparse as scs
import re
# from meshor.mesh import Mesh
from tqdm import tqdm
import re
from dgl.geometry import farthest_point_sampler
import potpourri3d as pp3d
import torch



# # sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/"))  # add the path to the DiffusionNet src
# import diffusion_net
# from diffusion_net import geometry
# from diffusion_net.utils import toNP

def toNP(x):
    """
    Really, definitely convert a torch tensor to a numpy array
    """
    return x.detach().to(torch.device('cpu')).numpy()


def read_plyfile(file_path, cls='o3d'):
    if cls == 'plyfile':
        plydata = PlyData.read(file_path)

        # extracting vertices, faces and color into arrays
        vertices = np.stack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).transpose()
        faces = np.stack(plydata['face']['vertex_indices'], axis=0)

        try:
            colors = np.stack(
                [plydata['vertex']['red'], plydata['vertex']['green'], plydata['vertex']['blue']]).transpose()
        except:
            colors = None

    elif cls == 'o3d':
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
        ValueError(f"'cls' argument should be one of: 'o3d', 'plyfile', 'pp3d'")

    return vertices, faces, colors


def read_ply2mesh(mesh_path, cls):
    vertices, faces, colors = read_plyfile(mesh_path)

    if cls == 'o3d':
        # creating open3d mesh
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
        mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(colors / 256)

        return mesh_o3d

    elif cls == 'pv':
        # creating pyvista object
        stack_vec = faces.shape[1] * np.ones((faces.shape[0], 1), dtype=int)
        mesh_pv = pv.PolyData(vertices, np.hstack((stack_vec, faces)))
        mesh_pv['colors'] = colors / 256

        return mesh_pv

    elif cls == 'o3d_pv':
        # creating open3d mesh
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
        mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(colors / 256)
        # creating pyvista object
        stack_vec = faces.shape[1] * np.ones((faces.shape[0], 1), dtype=int)
        mesh_pv = pv.PolyData(vertices, np.hstack((stack_vec, faces)))
        mesh_pv['colors'] = colors / 256

        return mesh_o3d, mesh_pv

    else:
        ValueError("Accepted objects: 'o3d' for Open3D or 'pv' for Pyvista or 'o3d_pv' for both")


def write_plyfile(filepath, vertices, faces, colors=None):
    mesh_o3d = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces))

    if colors is not None:
        mesh_o3d.vertex_colors = colors
        colors_flag = True
    else:
        colors_flag = False

    o3d.io.write_triangle_mesh(filename=filepath, mesh=mesh_o3d, write_ascii=True, write_vertex_colors=colors_flag)


def vis_pcd(vertices, faces=None, title=None, scalar_func=None, plotter=None, plot_location=None, point_size=5.,
            render_points_as_spheres=True,
            sphere=False, show_edges=False):
    if scalar_func is None:
        scalar_func = vertices[:, 2]

    if not show_edges:
        pv_object = pv.PolyData(vertices)
    else:
        stack_vec = faces.shape[1] * np.ones((faces.shape[0], 1), dtype=int)
        pv_object = pv.PolyData(vertices, np.hstack((stack_vec, faces)))

    if plotter is None:
        flag_plotter = True
        plotter = pv.Plotter()
    else:
        flag_plotter = False
        plotter.subplot(plot_location[0], plot_location[1])

    plotter.add_mesh(pv_object, style='points', point_size=point_size, scalars=scalar_func, cmap="rainbow",
                     render_points_as_spheres=render_points_as_spheres, show_edges=show_edges)
    if sphere:
        plotter.add_mesh(pv.Sphere(radius=0.5), opacity=0.7, color='red')

    if title is not None:
        plotter.add_title(title)

    if flag_plotter:
        plotter.show()


def vis_wireframe(vertices, faces, plotter=None, plot_location=None, title=None, sphere=False):
    stack_vec = faces.shape[1] * np.ones((faces.shape[0], 1), dtype=int)
    pv_object = pv.PolyData(vertices, np.hstack((stack_vec, faces)))

    if plotter is None:
        flag_plotter = True
        plotter = pv.Plotter(notebook=False)
    else:
        flag_plotter = False
        plotter.subplot(plot_location[0], plot_location[1])

    if title is not None:
        plotter.add_title(title)

    plotter.add_mesh(pv_object, style='wireframe')
    if sphere:
        plotter.add_mesh(pv.Sphere(radius=1), opacity=0.7, color='red')

    if flag_plotter:
        plotter.show()


def vis_surface(vertices, faces, scalar_func=None, plotter=None, plot_location=None, title=None, rgb=None,
                sphere=False, clim=None, show_scalar_bar=False):
    if scalar_func is None:
        scalar_func = vertices[:, 2]

    stack_vec = faces.shape[1] * np.ones((faces.shape[0], 1), dtype=int)
    pv_object = pv.PolyData(vertices, np.hstack((stack_vec, faces)))

    if plotter is None:
        flag_plotter = True
        plotter = pv.Plotter()
    else:
        flag_plotter = False
        plotter.subplot(plot_location[0], plot_location[1])

    if rgb is not None:
        pv_object['colors'] = rgb / 256
        plotter.add_mesh(pv_object, show_edges=False, scalars='colors', rgb=True, preference='point')
    elif clim is None:
        plotter.add_mesh(pv_object, style='surface', scalars=scalar_func, colormap='rainbow',
                         show_scalar_bar=show_scalar_bar)
    else:
        plotter.add_mesh(pv_object, style='surface', scalars=scalar_func, colormap='rainbow',
                         clim=clim, show_scalar_bar=show_scalar_bar)

    if sphere:
        plotter.add_mesh(pv.Sphere(radius=1), opacity=0.7, color='red')

    if title is not None:
        plotter.add_title(title)

    if flag_plotter:
        plotter.show()


def scale_save_model(mesh_path, scale_factor):
    plydata = PlyData.read(mesh_path)

    vertices = np.stack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).transpose()

    vertices3 = center_model(scale_model(vertices, factor=scale_factor))

    plydata['vertex']['x'] = vertices3[:, 0]
    plydata['vertex']['y'] = vertices3[:, 1]
    plydata['vertex']['z'] = vertices3[:, 2]

    main_path, file_name = os.path.split(mesh_path)
    name, ext = os.path.splitext(file_name)
    new_mesh_path = pjoin(main_path, f"{name}_scaled{scale_factor}{ext}")

    plydata.write(new_mesh_path)

    print(f"New scaled model was saved into: {new_mesh_path}")


def center_model(vertices):
    return vertices - vertices.mean(axis=0)


def scale_model(vertices, factor):
    return vertices * factor


def create_poly_data(vertices, faces=None, colors=None):
    if faces is not None:
        stack_vec = faces.shape[1] * np.ones((faces.shape[0], 1), dtype=int)
        pv_object = pv.PolyData(vertices, np.hstack((stack_vec, faces)))

    else:
        pv_object = pv.PolyData(vertices)

    if colors is not None:
        pv_object['colors'] = colors / 256

    return pv_object


def get_closest_points(vertices, points):
    closest_points_ind = []
    for point_ii in points[:]:
        idx = LA.norm(vertices - point_ii, axis=1).argmin()
        closest_points_ind.append(idx)
    return closest_points_ind


def rows_uniq_elems(a):
    b = np.array([v for v in a if len(set(v)) == len(v)])
    c = np.unique(np.sort(b), axis=0)

    return c


def compute_inlier_vertices_gdist(mesh, annotaion_indices, smaller_factor=1):
    vertices, faces = mesh.v, mesh.f

    # distance between annotated points and themselves - aiming to find the maximal distance
    gd = gdist.distance_matrix_of_selected_points(vertices=vertices.astype(np.float64), triangles=faces,
                                                  points=np.array(annotaion_indices, dtype=np.int32))

    # distance between annotated points to all points on the mesh
    gd_points_mesh = []
    for point_ii in annotaion_indices:
        gd_point = gdist.compute_gdist(vertices=vertices.astype(np.float64), triangles=faces,
                                       source_indices=np.array([point_ii], dtype=np.int32))
        gd_points_mesh.append(gd_point)
    gd_points_mesh = np.array(gd_points_mesh)

    # find the points on the mesh which the distance between them and the annotated points is smaller than the maximum
    smaller_than_max_bool = np.all(gd_points_mesh <= smaller_factor * gd.max(), axis=0)
    smaller_than_max_ind = np.where(np.all(gd_points_mesh <= smaller_factor * gd.max(), axis=0))[0]
    relevent_vertices = vertices[smaller_than_max_ind]

    return relevent_vertices, smaller_than_max_ind, smaller_than_max_bool, gd_points_mesh


def compute_inlier_vertices_gdist_centered(mesh, annotaion_indices, central_ind=None):
    vertices, faces = mesh.v, mesh.f
    annotaion_indices = np.array(annotaion_indices, dtype=np.int32)
    if central_ind is None:
        # find central point
        mean_value = np.array([np.mean(mesh.v[annotaion_indices], axis=0)])
        central_ind = np.array(get_closest_points(vertices=mesh.v, points=mean_value))

    # points which their gdist from all annotations is smaller than the maximum gdist of the annotation points
    gdist_vertices, gdist_ind, gdist_bool, gd_points_mesh = compute_inlier_vertices_gdist(
        mesh=mesh, annotaion_indices=annotaion_indices, smaller_factor=1)

    # distance between centered point and annotated points
    gd_center_anno = gdist.compute_gdist(vertices=vertices.astype(np.float64), triangles=faces,
                                         source_indices=np.array(central_ind, dtype=np.int32),
                                         target_indices=annotaion_indices)

    # distance between centered point and all gdist points
    gd_center_gdist = gdist.compute_gdist(vertices=vertices.astype(np.float64), triangles=faces,
                                          source_indices=np.array(central_ind, dtype=np.int32),
                                          target_indices=gdist_ind.astype(np.int32))

    # closest distance of all gdist points to annotated points (with it's index) - all points on mesh
    closest_anno_point_dist = np.vstack((gd_points_mesh.min(axis=0), gd_points_mesh.argmin(axis=0)))

    # closest distance of all gdist points to annotated points (with it's index) - only the gdist segment
    closest_anno_point_dist_tmp = closest_anno_point_dist[:, gdist_ind]

    gdist_centered_ind = gdist_ind[
        np.where(gd_center_gdist < gd_center_anno[closest_anno_point_dist_tmp[1].astype(np.int32)])[0]]

    return gdist_centered_ind, gdist_ind


def cut_path_segment(mesh, points_ind, central_ind=None):
    if central_ind is None:
        # find central point
        mean_value = np.array([np.mean(mesh.v[points_ind], axis=0)])
        central_ind = np.array(get_closest_points(vertices=mesh.v, points=mean_value))

    # split into source and target for path finder
    points_ind_tmp = np.append(points_ind, points_ind[0])
    n_indices = points_ind_tmp.shape[0]
    indices_split = np.array([list(points_ind_tmp[i: i + 2]) for i in range(0, n_indices - 1, 1)])

    # find edges according the vertex-vertex adjacency matrix
    adj_vv_mat = mesh.vertex_vertex_adjacency()
    edges_raw = scs.find(adj_vv_mat)
    edges = rows_uniq_elems(np.array([edges_raw[0], edges_raw[1]]).T)
    # calculate edges length
    edges_length = LA.norm(mesh.v[edges[:, 0]] - mesh.v[edges[:, 1]], axis=1)

    # create nx graph
    G = nx.from_scipy_sparse_array(adj_vv_mat)
    length_dict = {edge_ii: {'length': length_ii} for edge_ii, length_ii in zip(G.edges(), edges_length)}
    nx.set_edge_attributes(G, length_dict)

    # compute paths between the annotated points
    points_path = []
    for ii, couple_ind_ii in enumerate(indices_split):
        sourceIndex, targetIndex = couple_ind_ii
        path = nx.shortest_path(G, source=sourceIndex, target=targetIndex, weight='length')
        if ii == indices_split.shape[0]:
            points_path = points_path + path
        else:
            points_path = points_path[0:-1] + path

    points_path = np.array(points_path)
    path_indices = np.unique(points_path)

    # find edges which part of the path
    n_path = len(points_path)
    path_edges = np.array([list(points_path[i: i + 2]) for i in range(0, n_path - 1, 1)])
    path_edges = rows_uniq_elems(path_edges)
    edges_path_indices = np.where((edges[None, :] == path_edges[:, None]).all(-1).any(0))[0]

    # dividing the segment from the mesh
    # the segment is isolated by erasing all the edges with the path vertices
    adj_vv_mat_tmp = adj_vv_mat.tolil(copy=True)
    adj_vv_mat_tmp[path_indices] = 0
    adj_vv_mat_tmp[:, path_indices] = 0
    adj_vv_mat_tmp = adj_vv_mat_tmp.tocsr()
    adj_vv_mat_tmp.eliminate_zeros()

    # the new Graph with the isolated segment
    G_new = nx.from_scipy_sparse_array(adj_vv_mat_tmp)
    # the component (segment) of the central point
    g_new_connect = np.array(list(nx.node_connected_component(G_new, central_ind[0])))

    # adding the path points to the segment
    # segment_ind = np.hstack((g_new_connect, path_indices))
    segment_ind = g_new_connect

    # find the faces which contains the segment points
    # segment_faces = np.where(faces == segment_ind[:, None][:, None])[1]

    return segment_ind, points_path, edges


def my_split(s):
    return list(filter(None, re.split(r'(\d+)', s)))


def get_large_mesh_component(mesh):
    """
    :param mesh:
    :return:
        vertices - the vertices of the mesh
        faces list (Nfaces, 3) - new list excluded the non connected components
    """

    # create a network from mesh
    G = nx.from_scipy_sparse_array(mesh.vertex_vertex_adjacency())

    # list of sub-graphs
    subgraph_list = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    subgraph_list_len = np.array([sub_ii.number_of_nodes() for sub_ii in subgraph_list])

    if subgraph_list_len.shape[0] == 1:
        print("There is only 1 connected components in this mesh\nReturning the input mesh")
        return mesh

    # delete the largest component since we want to exlode all the others
    del subgraph_list[subgraph_list_len.argmax()]

    # list of all unwanted vertices
    unnode_list = np.array(sum([list(graph_ii.nodes()) for graph_ii in subgraph_list], []))

    # find the faces which are stays relevant
    unfaces_list = np.where(mesh.f == unnode_list[:, None][:, None])[1]
    faces_list = np.delete(mesh.f, obj=unfaces_list, axis=0)

    #
    new_vertices = np.delete(mesh.v, obj=unnode_list, axis=0)
    innode_list = np.delete(np.arange(mesh.v.shape[0]), obj=unnode_list, axis=0)
    newnode_list = np.arange(innode_list.shape[0])

    new_faces = faces_list.copy()
    print(f"Updating faces array, it will take a while")

    for ii, (in_ii, new_ii) in tqdm(enumerate(zip(innode_list, newnode_list)), total=innode_list.shape[0]):
        new_faces[faces_list == in_ii] = new_ii

    new_colors = mesh.c[innode_list]

    new_mesh = Mesh(vertices=new_vertices, faces=new_faces, colors=new_colors)

    return new_mesh


def remove_species_from_text(label, comments):
    """

    :param label: label string example: "acropora_tabular_08"
                    pattern:

                    <species>_<morphology>_<morphology ID>

                    or
                    <species> <morphology> <morphology ID>

                    or
                    <morphology> <morphology ID>

                    or
                    <morphology><morphology ID>

    :return:  <species> , <morphology>_<morphology ID>
    <species>=None when no species information
    """

    # m = re.search(r"\d", label)
    # text_digit_list = label[0:m.end() + 1].split()
    # if len(text_digit_list) > 2:
    #     label_list = label.split()[1:]
    #     return ' '.join(word for word in label_list)
    # else:
    #     return label
    a = 10

    # split the comment from the label

    comment = [comment_ii for comment_ii in comments if comment_ii in label]
    if comment:
        comment = comment[0]
        label_update = my_split(label)[0] + my_split(label)[1]
    else:
        comment = 'point'
        label_update = label

    # method 1
    label_parts = label_update.split("_")
    if len(label_parts) == 3:
        species = label_parts[0]
        morpho = label_parts[1] + label_parts[2]

    elif len(label_parts) == 1:
        label_parts = label_update.split(" ")
        if '' in label_parts:
            label_parts = [label_part_ii for label_part_ii in label_parts if label_part_ii]

        # method 2
        if len(label_parts) == 3:
            species = label_parts[0]
            morpho = label_parts[1] + label_parts[2]

        # method 3
        elif len(label_parts) == 2:
            species = float("Nan")
            morpho = label_parts[0] + label_parts[1]

        # method 4
        else:
            species = float("Nan")
            morpho = label_update

    return species, morpho, comment


def farthest_points_sample(vertices, n_samples, start_idx=0, faces=None, colors=None, visualize=False):
    """

    :param vertices: vertices of the mesh
    :param faces: faces of the mesh
    :param n_samples: number of sampled points
    :param start_idx: the index of the starting sampling
    :param colors: color per vertex
    :return: the indices of the sampled vertices anf the vertices of the patch
    """

    fps_idx = torch.squeeze(
        farthest_point_sampler(torch.unsqueeze(torch.Tensor(vertices), 0), n_samples, start_idx=start_idx)).numpy()

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


def create_3D_patch(vertices, point_index, faces=None, edges=None, labels={}, t_coef=0.05, n_patch_vertices=10000,
                    heat_distance_solver=None, colors=None, visualize=False):
    """

    creating a patch of a point cloud or triangle mesh
    if there are vertex/face/edge labels - it is also calculated

    :param vertices: mesh vertices
    :param point_index: index of the vertex for the patch
    :param faces: mesh faces
    :param edges: mesh edges
    :param labels: dictionary of the vertex, face and edge labels
    :param t_coef: time value for the heat distance calculation
    :param n_patch_vertices: number of vertices for the patch (the closest to the point)
    :param colors: in case colors are exist, only for visualization reasons
    :param visualize: True/False for patch visualization
    :return: vertices[N,3]
            faces[M,3]
            colors[N, 3]
            labels (dictionary of vertex, face and edge labels) of the patch
            original_idx - 'vertices', 'faces' dictionary of original vertices idx

    """

    # TODO:
    #   1. adding the GT segments for the patch (after deciding what are the annotations - edges/points/faces)
    #   2. vertices colors

    # solver for het method distance by N.Sharp
    if heat_distance_solver is None:
        heat_distance_solver = pp3d.MeshHeatMethodDistanceSolver(vertices, faces, t_coef=t_coef)

    # find a distance from some point
    dist = heat_distance_solver.compute_distance(point_index)

    # dictionary for the original vertices indices
    original_idx = {}
    # dictionary for 'vertex', 'face' and 'edge' patch label - will stay empty if there are no input labels
    labels_patch = {}

    # find patch vertices
    vertices_patch_idx = dist.argsort()[:n_patch_vertices]
    original_idx['vertices'] = vertices_patch_idx
    vertices_patch = vertices[vertices_patch_idx]

    if colors is not None:
        colors_patch = colors[vertices_patch_idx]
    else:
        colors_patch = None

    if 'vertex' in labels.keys():
        labels_patch['vertex'] = labels['vertex'][vertices_patch_idx]

    # find patch faces - only if have it in the input
    if faces is not None:

        # patch faces indices
        find_faces_idx = np.where(np.isin(faces, vertices_patch_idx).sum(axis=1) == 3)[0]
        faces_patch_tmp = faces[find_faces_idx]
        faces_patch = faces[find_faces_idx]
        # attaching the corresponded edges for the vertices (according the updated indices of the vertices)
        for ii, item in enumerate(vertices_patch_idx):
            faces_patch[faces_patch_tmp == item] = ii

        # remembering the original indices for assembling the mesh
        original_idx['faces'] = faces[find_faces_idx]

        # if there are face labels, attach it
        if 'face' in labels.keys():
            labels_patch['face'] = labels['face'][find_faces_idx]

        # if there are faces as input let get the edges data too
        # find patch edges
        if edges is None:
            edges, _ = get_mesh_edges(vertices=vertices, faces=faces)
        find_edges_idx = np.where(np.isin(edges, vertices_patch_idx).sum(axis=1) == 2)[0]
        edges_patch_tmp = edges[find_edges_idx]
        edges_patch = edges[find_edges_idx]

        # attaching the corresponded edges for the vertices (according the the updated indices of the vertices)
        for ii, item in enumerate(vertices_patch_idx):
            edges_patch[edges_patch_tmp == item] = ii

        # remembering the original indices for assembling the mesh
        original_idx['edges'] = edges[find_edges_idx]

        # if there are edge labels, attach it
        if 'edge' in labels.keys():
            labels_patch['edge'] = labels['edge'][find_edges_idx]

    # visualization of the patch on the mesh
    if visualize:
        plotter = pv.Plotter()
        if colors is None:
            plotter.add_mesh(create_poly_data(vertices, faces), scalars=vertices[:, 2], colormap='rainbow')
        else:
            plotter.add_mesh(create_poly_data(vertices, faces, colors=colors),
                             scalars='colors', rgb=True, preference='point')

        plotter.add_mesh(create_poly_data(vertices_patch, faces_patch), color='#C875C4', show_edges=False)

        plotter.add_mesh(create_poly_data(vertices[point_index]), style='points', color='red', point_size=15.,
                         render_points_as_spheres=True, show_scalar_bar=False)
        plotter.show()

    return vertices_patch, faces_patch, edges_patch, colors_patch, labels_patch, original_idx


def get_mesh_edges(vertices, faces):
    """

    :param vertices: mesh vertices
    :param faces: mesh faces
    :return: mesh edges indices of the vertices [N edges, 2]
    """

    edges, edges_count = np.unique(np.sort(np.concatenate((faces[:, 0:2], faces[:, 1:], faces[:, [0, 2]]))), axis=0,
                                   return_counts=True)

    return edges, edges_count


def list_diff(li1, li2):
    li_dif = [i for i in li1 + li2 if i not in li1 or i not in li2]
    return li_dif


# %% create path if not exists

def new_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

    return path
