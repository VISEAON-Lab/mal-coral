import numpy as np
from numpy import linalg as LA
import scipy.sparse as scs
from scipy.sparse import csr_matrix, identity, diags
import networkx as nx

def get_rows_norm(mat):
    l2_norm = np.linalg.norm(mat, axis=1)
    return l2_norm

def get_face_normals(v, f, normalized=True):
    a = v[f[:, 0], :]
    b = v[f[:, 1], :]
    c = v[f[:, 2], :]
    fn = np.cross(b - a, c - a)
    if normalized:
        norms = get_rows_norm(fn)
        fn = fn / norms[:, None]
    return fn

def get_face_barycenters(v, f):
    face_coordinates = v[f[:], :]
    face_centers = np.mean(face_coordinates, axis=1)
    return face_centers

def get_face_areas(v, f):
    fn = get_face_normals(v, f, normalized=False)
    fn_magnitude = np.linalg.norm(fn, axis=1)
    face_areas = fn_magnitude / 2
    return face_areas


# %% mesh adjacency functions
def vertex_face_adjacency(vertices, faces):
    num_vertices = vertices.shape[0]
    num_faces = faces.shape[0]
    face_len = faces.shape[1]
    rows = np.concatenate(list(faces))
    cols = np.arange(num_faces * face_len) // face_len
    assert np.shape(rows) == np.shape(cols)
    data = np.ones_like(rows)
    vf_adj = csr_matrix((data, (rows, cols)), shape=(num_vertices,
                                                     num_faces), dtype=bool)
    return vf_adj


def vertex_vertex_adjacency(vertices, faces):
    vf_adj = vertex_face_adjacency(vertices, faces)
    vv_mat = vf_adj.dot(vf_adj.transpose())
    vv_adj = (vv_mat - identity(len(vertices)))
    return vv_adj


# %% other functions
def rows_uniq_elems(a):
    b = np.array([v for v in a if len(set(v)) == len(v)])
    c = np.unique(np.sort(b), axis=0)
    return c


def get_closest_points(vertices, points):
    closest_points_ind = []
    for point_ii in points[:]:
        idx = LA.norm(vertices - point_ii, axis=1).argmin()
        closest_points_ind.append(idx)
    return closest_points_ind


# %% cut mesh path function

def cut_path_segment(vertices, faces, polygons_points_ind, centrals_ind=None, group=None):
    # print('=======================================================================')
    print('======= find edges according the vertex-vertex adjacency matrix =======')
    # print('=======================================================================')
    # find edges according the vertex-vertex adjacency matrix
    adj_vv_mat = vertex_vertex_adjacency(vertices, faces)
    edges_raw = scs.find(adj_vv_mat)
    edges = rows_uniq_elems(np.array([edges_raw[0], edges_raw[1]]).T)
    # calculate edges length
    edges_length = LA.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)

    # print('===========================================')
    print('============= create nx graph =============')
    # print('===========================================')
    # create nx graph
    G = nx.from_scipy_sparse_matrix(adj_vv_mat)
    length_dict = {edge_ii: {'length': length_ii} for edge_ii, length_ii in zip(G.edges(), edges_length)}
    nx.set_edge_attributes(G, length_dict)

    print('========================================================')
    print('====== compute paths between the annotated points ======')
    print('========================================================')

    # split into source and target for path finder
    paths = []
    for idx, polygon_points_ind in enumerate(polygons_points_ind):
        print(f'====== compute paths for segment {idx} ======')
        path = {}
        if group is not None:
            #the group of the segment
            path['group'] = group[idx]
        #the segment surrounding vertices (only the annotated ones, first vertex repeats)
        path["annotated_vertex_ind"] = np.append(polygon_points_ind, polygon_points_ind[0])
        # number of annotated vertices + 1
        n_indices = path["annotated_vertex_ind"].shape[0]
        #splits the annotated vertices into n_indices - 1 pairs (each vertex in two pairs)
        indices_split = np.array(
            [list(path["annotated_vertex_ind"][i: i + 2]) for i in range(0, n_indices - 1, 1)])

        # compute paths between the annotated points
        points_path = []
        for ii, couple_ind_ii in enumerate(indices_split):
            sourceIndex, targetIndex = couple_ind_ii
            tmp_path = nx.shortest_path(G, source=sourceIndex, target=targetIndex, weight='length')
            if ii == indices_split.shape[0]:
                points_path = points_path + tmp_path
            else:
                points_path = points_path[0:-1] + tmp_path

        # contains all the vertices surrounding the segment, not just the annotated ones, not just the unique ones
        path["surrounding_vertex_ind"] = np.unique(points_path) #path_indices

        paths.append(path)

    print('============= dividing segments from the mesh =============')
    # the segment is isolated by erasing all the edges with the path vertices
    adj_vv_mat_tmp = adj_vv_mat.tolil(copy=True)

    total_path_indices = np.unique(np.concatenate([path["surrounding_vertex_ind"] for path in paths]))
    adj_vv_mat_tmp[total_path_indices] = 0
    adj_vv_mat_tmp[:, total_path_indices] = 0

    adj_vv_mat_tmp = adj_vv_mat_tmp.tocsr()
    adj_vv_mat_tmp.eliminate_zeros()

    print('============= create new nx graph =============')
    # the new Graph with the isolated segment
    G_new = nx.from_scipy_sparse_matrix(adj_vv_mat_tmp)

    for path, central_ind in zip(paths, centrals_ind):
        #all the vertices inside the segment
        g_new_connect = np.array(list(nx.node_connected_component(G_new, central_ind)))

        #all the vertices around and inside the segment
        #path['segment_vertex_ind'] = g_new_connect old
        path['segment_vertex_ind'] = np.hstack((g_new_connect, path['surrounding_vertex_ind']))

        # find the faces which contains the segment points (face is labeled when all the 3 vertices are labeled)
        path['segment_face_idx'] = np.where(np.isin(faces, path['segment_vertex_ind']).sum(axis=1) == 3)[0]

    return paths
