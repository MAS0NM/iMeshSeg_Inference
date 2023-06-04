from utils import read_dir, read_filenames, filelist_checker
from preprocessing import *
from os import cpu_count

if __name__ == '__main__':
    dir_paths = ['./dataset/3D_scans_per_patient_obj_files_b1', './dataset/3D_scans_per_patient_obj_files_b2']
    # label diractory in those 2 batches
    label_paths = [dir_path + '/ground-truth_labels_instances_b' + str(idx+1) for idx, dir_path in enumerate(dir_paths)]
    # file list path
    fileLists_path = './dataset/FileLists'
    downsampled_dataset = './dataset/test_set/'

    print('number of cpus', cpu_count())
    lower_jaws, upper_jaws = read_filenames(dir_paths)
    lower_labels, upper_labels = read_filenames(label_paths)
    lower_jaws, lower_labels = filelist_checker(lower_jaws, lower_labels)
    upper_jaws, upper_labels = filelist_checker(upper_jaws, upper_labels)
    fileList_lower = '/fileList_lower.txt'
    fileList_upper = '/fileList_upper.txt'
    
    target_cells = 10001
    existing_mesh_files = read_dir(dir_path=downsampled_dataset, extension='vtk', constrain='')
    do_downsample(upper_jaws, upper_labels, target_cells=target_cells, ds_dir = downsampled_dataset)
    print(f"downsampling is done")
    do_augmentation(ip_dir=downsampled_dataset, op_dir=downsampled_dataset, aug_num=1, existing_mesh_files=existing_mesh_files)
    print(f"augmentation is done")
    