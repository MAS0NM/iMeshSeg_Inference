import numpy as np
import os        
import vedo
import json
import glob
from collections import Counter
from tqdm import tqdm
from multiprocessing import cpu_count
from joblib import Parallel, delayed
from utils import get_sample_name
import vtk

            
def centring(mesh):
    mesh.points(pts=mesh.points()-mesh.center_of_mass())
    return mesh


def set_face_label(mesh, labels_by_point, mode='count'):
    labels_by_face = []
    if mode == 'count':
        for face in mesh.faces():
            vertex_labels = [labels_by_point[i] if i < len(labels_by_point) else 0 for i in face]
            if all(x == vertex_labels[0] for x in vertex_labels):
                # print(face, vertex_labels)
                labels_by_face.append(vertex_labels[0])
            else:
                label_count = Counter(vertex_labels)
                most_common_label = label_count.most_common(1)[0][0]
                labels_by_face.append(most_common_label)
    elif mode == 'distance':
        for face in mesh.faces():
            vertex_labels = [labels_by_point[i] for i in face]
            vertices = mesh.points()[face]
            centroid = np.array([sum([x[i] for x in vertices])/3 for i in range(3)], dtype=np.uint8)
            distances = np.array([sum(np.square(vertices-centroid))], dtype=np.float16)
            nearest_point_index = np.argmin(distances)
            this_label = vertex_labels[nearest_point_index]
            labels_by_face.append(this_label)
    return labels_by_face

            
def rearrange(nparry):
    # 32 permanent teeth
    nparry[nparry == 17] = 1
    nparry[nparry == 37] = 1
    nparry[nparry == 16] = 2
    nparry[nparry == 36] = 2
    nparry[nparry == 15] = 3
    nparry[nparry == 35] = 3
    nparry[nparry == 14] = 4
    nparry[nparry == 34] = 4
    nparry[nparry == 13] = 5
    nparry[nparry == 33] = 5
    nparry[nparry == 12] = 6
    nparry[nparry == 32] = 6
    nparry[nparry == 11] = 7
    nparry[nparry == 31] = 7
    nparry[nparry == 21] = 8
    nparry[nparry == 41] = 8
    nparry[nparry == 22] = 9
    nparry[nparry == 42] = 9
    nparry[nparry == 23] = 10
    nparry[nparry == 43] = 10
    nparry[nparry == 24] = 11
    nparry[nparry == 44] = 11
    nparry[nparry == 25] = 12
    nparry[nparry == 45] = 12
    nparry[nparry == 26] = 13
    nparry[nparry == 46] = 13
    nparry[nparry == 27] = 14
    nparry[nparry == 47] = 14
    nparry[nparry == 18] = 15
    nparry[nparry == 38] = 15
    nparry[nparry == 28] = 16
    nparry[nparry == 48] = 16
    # deciduous teeth
    nparry[nparry == 55] = 3
    nparry[nparry == 55] = 3
    nparry[nparry == 54] = 4
    nparry[nparry == 74] = 4
    nparry[nparry == 53] = 5
    nparry[nparry == 73] = 5
    nparry[nparry == 52] = 6
    nparry[nparry == 72] = 6
    nparry[nparry == 51] = 7
    nparry[nparry == 71] = 7
    nparry[nparry == 61] = 8
    nparry[nparry == 81] = 8
    nparry[nparry == 62] = 9
    nparry[nparry == 82] = 9
    nparry[nparry == 63] = 10
    nparry[nparry == 83] = 10
    nparry[nparry == 64] = 11
    nparry[nparry == 84] = 11
    nparry[nparry == 65] = 12
    nparry[nparry == 85] = 12
    
    return nparry


def flip_relabel(nparry):
    # 1 14, 2 13, 3 12, 4 11, 5 10, 6 9, 7 8, 15 16
    pairs = [(1, 14), (2, 13), (3, 12), (4, 11), (5, 10), (6, 9), (7, 8), (15, 16)]
    for x, y in pairs:
        index_x = np.where(nparry == x)
        index_y = np.where(nparry == y)
        np.put(nparry, index_x, y)
        np.put(nparry, index_y, x)
        
    return nparry


def downsample_with_index(mesh, target_cells):
    mesh_ds = mesh.clone()
    mesh_ds = mesh_ds.decimate(target_cells / mesh.ncells)
    indices = np.array([mesh.closest_point(i, return_point_id=True) for i in mesh_ds.points()])
    if mesh_ds.ncells > target_cells - 1:
        for i in range(mesh_ds.ncells):
            mesh_cp = mesh_ds.clone()
            mesh_cp.delete_cells([i])
            if mesh_cp.ncells == target_cells - 1:
                return mesh_cp, indices
        
    return mesh_ds, indices


def downsample(jaw_path, lab_path, target_cells, op_dir, all_file_list):
    '''
        down sample and store in vtk form with celldata['labels'] and pointdata['labels']
    '''
    sampleName = get_sample_name(jaw_path)
    path_mesh = op_dir + sampleName + '.vtk'
    
    if sampleName in all_file_list:
        return
    
    mesh = vedo.load(jaw_path)
    mesh = centring(mesh)
            
    # Add cell data with labels
    with open(lab_path, 'r') as f:
        labels_by_point = np.array(json.load(f)['labels'], dtype=np.uint8)
        labels_by_point = rearrange(labels_by_point)
    # raw mesh
    # labels_by_face = set_face_label(mesh, labels_by_point, mode='count')
    # mesh.pointdata['labels'] = labels_by_point
    # mesh.celldata['labels'] = labels_by_face
    
    # downsample
    try:
        mesh_ds, indices = downsample_with_index(mesh, target_cells)
        labels_by_point_ds = labels_by_point[indices].tolist()
        labels_by_face_ds = set_face_label(mesh_ds, labels_by_point_ds, mode='count')
        mesh_ds.pointdata['labels'] = labels_by_point_ds
        mesh_ds.celldata['labels'] = labels_by_face_ds
    except:
        print(jaw_path, lab_path, 'error', mesh.npoints, labels_by_point.shape)
        # visualize_mesh(mesh)
        return
    
    # write mesh vtk
    mesh_ds.write(path_mesh)


def do_downsample(jaws, labels, target_cells, ds_dir = './dataset/test_set/'):
    # get all files in 3d_scans_ds
    all_ds_files = glob.glob(os.path.join(ds_dir, "*"))
    all_ds_files = [get_sample_name(path) for path in all_ds_files]
    Parallel(n_jobs=cpu_count())(delayed(downsample)(jaw, lab, target_cells, ds_dir, all_ds_files) for jaw, lab in tqdm(iterable=list(zip(jaws, labels)), desc='down sampling'))
   

def GetVTKTransformationMatrix(rotate_X=[-180, 180], rotate_Y=[-180, 180], rotate_Z=[-180, 180],
                                translate_X=[-10, 10], translate_Y=[-10, 10], translate_Z=[-10, 10],
                                scale_X=[0.8, 1.2], scale_Y=[0.8, 1.2], scale_Z=[0.8, 1.2],
                                isRotate=True, isTranslate=True, isScale=True):
    # create a vtk transform object
    transform = vtk.vtkTransform()
    # set random rotation angles
    if isRotate:
        if type(rotate_X) == list and type(rotate_Y) == list and type(rotate_Z) == list:
            angle_x = np.random.uniform(rotate_X[0], rotate_X[1])
            angle_y = np.random.uniform(rotate_Y[0], rotate_Y[1])
            angle_z = np.random.uniform(rotate_Z[0], rotate_Z[1])
            transform.RotateX(angle_x)
            transform.RotateY(angle_y)
            transform.RotateZ(angle_z)
        elif type(rotate_X) == int and type(rotate_Y) == int and type(rotate_Z) == int:
            transform.RotateX(rotate_X)
            transform.RotateY(rotate_Y)
            transform.RotateZ(rotate_Z)
            
    # set random translation distances
    if isTranslate:
        dist_x = np.random.uniform(translate_X[0], translate_X[1])
        dist_y = np.random.uniform(translate_Y[0], translate_Y[1])
        dist_z = np.random.uniform(translate_Z[0], translate_Z[1])
        transform.Translate(dist_x, dist_y, dist_z)
    # set random scaling factors
    if isScale:
        factor_x = np.random.uniform(scale_X[0], scale_X[1])
        factor_y = np.random.uniform(scale_Y[0], scale_Y[1])
        factor_z = np.random.uniform(scale_Z[0], scale_Z[1])
        transform.Scale(factor_x, factor_y, factor_z)
    # get the transformation matrix
    matrix = transform.GetMatrix()
    return matrix
        
        
def augment(mesh_path, out_dir, mode, aug_num, existing_mesh_files, isRotate, isTranslate, isScale, ext='.vtk'):
    if 'AUG' in mesh_path:
        return
    mesh_name = get_sample_name(mesh_path)
    if mode == 'flip':
        if 'FLP' in mesh_path:
            return
        mesh_op_pth = os.path.join(out_dir, mesh_name + '_FLP' + ext)
        if mesh_op_pth in existing_mesh_files:
            # print(f'{mesh_path} exists, skip')
            return
        mesh = vedo.load(mesh_path)
        mesh.mirror(axis='x')
        mesh.celldata['labels'] = flip_relabel(mesh.celldata['labels'])
        mesh = centring(mesh)
        mesh.write(mesh_op_pth)
    else:
        for i in range(aug_num):
            transform_type = ''
            if isRotate:
                transform_type += 'R'
            if isTranslate:
                transform_type += 'T'
            if isScale:
                transform_type += 'S'
            mesh_op_pth = os.path.join(out_dir, mesh_name+'_'+transform_type+"_AUG%02d_" %i + ext)
            if mesh_op_pth in existing_mesh_files:
                print(f'{mesh_path} exists, skip')
                continue
            vtk_matrix = GetVTKTransformationMatrix(rotate_X=[-180, 180], rotate_Y=[-180, 180], rotate_Z=[-180, 180],
                                                    translate_X=[-10, 10], translate_Y=[-10, 10], translate_Z=[-10, 10],
                                                    scale_X=[0.8, 1.2], scale_Y=[0.8, 1.2], scale_Z=[0.8, 1.2],
                                                    isRotate=isRotate, isTranslate=isTranslate, isScale=isScale) #use default random setting
        
            mesh = vedo.load(mesh_path)
            mesh.apply_transform(vtk_matrix)
            mesh = centring(mesh)
            mesh.write(mesh_op_pth)
            
        
def do_augmentation(ip_dir, op_dir, aug_num, existing_mesh_files, ext='.vtk', filelist=None):
    '''
        this function will first look into the ip_dir to make a filelist according to the files in there
        then perform augmentation on each .vtk file
    '''
    # filp the meshes to double the size
    if filelist:
        filels = [i for i in filelist if i]
    else:
        filels = glob.glob(f"{ip_dir}/*.vtk")
        
    Parallel(n_jobs=cpu_count())(delayed(augment)(item, op_dir, 'flip', aug_num, existing_mesh_files, False, False, False, ext) for item in tqdm(filels, desc="flipping"))
    # random tranform
    Parallel(n_jobs=cpu_count())(delayed(augment)(item, op_dir, 'trsf', aug_num, existing_mesh_files, True, False, False, ext) for item in tqdm(filels, desc="rotating"))
    Parallel(n_jobs=cpu_count())(delayed(augment)(item, op_dir, 'trsf', aug_num, existing_mesh_files, False, True, False, ext) for item in tqdm(filels, desc="translating"))
    Parallel(n_jobs=cpu_count())(delayed(augment)(item, op_dir, 'trsf', aug_num, existing_mesh_files, False, False, True, ext) for item in tqdm(filels, desc="rescaling"))
    

def make_labeled_mesh(jaw_path, lab_path, op_dir):
    sampleName = get_sample_name(jaw_path)
    path_mesh = op_dir + sampleName + '.vtk'
    
    mesh = vedo.load(jaw_path)
    print(mesh.ncells)
            
    # Add cell data with labels
    with open(lab_path, 'r') as f:
        labels_by_point = np.array(json.load(f)['labels'], dtype=np.uint8)
        labels_by_point = rearrange(labels_by_point)
    # raw mesh
    labels_by_face = set_face_label(mesh, labels_by_point, mode='count')
    mesh.pointdata['labels'] = labels_by_point
    mesh.celldata['labels'] = labels_by_face
        
    # write mesh vtk
    mesh.write(path_mesh)