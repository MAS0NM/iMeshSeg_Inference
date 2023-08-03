from utils import read_dir, read_filenames, vtk2stl
from preprocessing import *
from os import cpu_count
import csv


def make_augmentated_stl(filelist, stl_dir):
    existing_mesh_files = read_dir(dir_path=stl_dir, extension='stl', constrain='')
    do_augmentation('', stl_dir, aug_num=1, existing_mesh_files=existing_mesh_files, filelist=filelist, ext='.stl')
    
    
def read_from_csv(file_path):
    samples = []
    labels = []
    with open(file_path, 'r') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            samples.append(row[0])
            labels.append(row[1])
    return samples, labels


def get_labels_by_sample(file_list, label_paths):
    lower_labels, upper_labels = read_filenames(label_paths)
    labels = lower_labels + upper_labels
    lookup_table = [get_sample_name(i) for i in labels]
    res = []
    for item in file_list:
        if get_sample_name(item) in lookup_table:
            res.append(item)
    return res


if __name__ == '__main__':
    dir_paths = ['./dataset/3D_scans_per_patient_obj_files_b1', './dataset/3D_scans_per_patient_obj_files_b2']
    # label diractory in those 2 batches
    label_paths = [dir_path + '/ground-truth_labels_instances_b' + str(idx+1) for idx, dir_path in enumerate(dir_paths)]
    # file list path
    fileLists_path = './dataset/FileLists'
    downsampled_dataset = './dataset/test_set/'

    print('number of cpus', cpu_count())
    # lower_jaws, upper_jaws = read_filenames(dir_paths)
    # lower_labels, upper_labels = read_filenames(label_paths)
    # lower_jaws, lower_labels = filelist_checker(lower_jaws, lower_labels)
    # upper_jaws, upper_labels = filelist_checker(upper_jaws, upper_labels)
    # fileList_lower = '/fileList_lower.txt'
    # fileList_upper = '/fileList_upper.txt'
    
    test_set_path = './tst_list.csv'
    test_set_samples, test_set_labels = read_from_csv(test_set_path)
    
    target_cells = 10001
    existing_mesh_files = read_dir(dir_path=downsampled_dataset, extension='vtk', constrain='')
    # do_downsample(upper_jaws[:40], upper_labels[:40], target_cells=target_cells, ds_dir = downsampled_dataset)
    # do_downsample(lower_jaws[:40], lower_labels[:40], target_cells=target_cells, ds_dir = downsampled_dataset)
    # do_downsample(test_set_samples, test_set_labels, target_cells=target_cells, ds_dir = downsampled_dataset)
    # print(f"downsampling is done")
    # do_augmentation(ip_dir=downsampled_dataset, op_dir=downsampled_dataset, aug_num=1, existing_mesh_files=existing_mesh_files)
    # print(f"augmentation is done")
    
    # make_augmentated_stl(upper_jaws[:40], './dataset/test_set_stl')
    
    
    # vtk2stl('./dataset/test_set/', './dataset/vtk2stl')