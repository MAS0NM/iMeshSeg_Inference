from utils import get_sample_name, recursively_get_file
import numpy as np
import vedo
from test_inf import infer
from omegaconf import OmegaConf
from model.LitModule import LitModule
from tqdm import tqdm
import json
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import random


def cal_mIoU(grtr, pred, cm=None):
    # grtr, pred = np.array(grtr), np.array(pred)
    # mIoU = (grtr & pred).sum() / (grtr | pred).sum()
    # return mIoU
    cm = confusion_matrix(grtr, pred) if not cm else cm
    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    # TN = cm.sum() - (FP + FN + TP)
    return np.mean(TP / (FN + FP + TP + np.finfo(float).eps)) #eps = 2**-52, used to avoid divided by zero


def cal_acc(grtr, pred):
    # grtr, pred = np.array(grtr), np.array(pred)
    # acc = np.sum(grtr == pred)
    # return acc / len(grtr)
    return accuracy_score(grtr, pred)


def read_into_dict(file_path):
    with open(file_path, 'r') as f:
        dic = {}
        for line in f.readlines():
            sample, grdt, pred = line.strip().split('\t')
            dic[sample] = {'ground_truth': grdt, 'predict_result': pred}
    return dic


def inf_on_one_sample(mesh_path, cfg, model, dic, with_new_features=False):
    sample_name = get_sample_name(mesh_path)
    if sample_name in dic.keys():
        return
    mesh_ds = vedo.Mesh(mesh_path)
    grdt = mesh_ds.celldata['labels']
    mesh_ds_pred = infer(cfg=cfg, model=model, mesh_file=mesh_ds, print_time=False, refine=True, with_raw_output=False, with_new_features=with_new_features)
    pred = mesh_ds_pred.celldata['labels']
    dic[sample_name] = {'ground_truth': grdt.tolist(), 'predict_result': pred.tolist()}
    

def do_inf(dataset_path, cfg_path, checkpoint_path, output_path, with_new_features=False):
    '''
        will get all the vtk files in dataset_path to do inference
        also check in the output json file, search by the sample name and skip those who's already predicted
    '''
    eval_list = recursively_get_file(dataset_path, ext='vtk')
    # samples = [item for item in eval_list if 'upper' in item][:200]
    random.shuffle(eval_list)
    samples = eval_list[:100]
    print(samples[:10])
    print(f'get {len(samples)} files for evaluation')
    
    cfg = OmegaConf.load(cfg_path)
    module = LitModule(cfg).load_from_checkpoint(checkpoint_path)
    model = module.model.to(device)
    model.eval()

    try:
        with open(output_path, 'r') as f:
            dic = json.load(f)
    except:
        dic = dict()
    
    for mesh_path in tqdm(samples, desc='inferencing'):
        inf_on_one_sample(mesh_path, cfg, model, dic, with_new_features)
        
    with open(output_path, "w") as f:
        json.dump(dic, f)
        

def eva_on_one_sample(item):
    sample_name = item[0]
    grth, pred = item[1]['ground_truth'], item[1]['predict_result']
    mIoU = cal_mIoU(grth, pred)
    acc = cal_acc(grth, pred)
    with open('evals.txt', 'a') as f:
        f.write(f'{sample_name}\tmIoU:{mIoU}\tacc:{acc}\n')
    return sample_name, mIoU, acc

        
def do_eva(pred_path, constrain=''):
    print(f'evaluation for {constrain}')
    with open(pred_path, 'r') as f:
        pred_res = json.load(f)
        
    with open('evals.txt', 'w'):
        pass
    
    orig, flip, rota, tran, scal = [], [], [], [], []
    
    for item in pred_res.items():
        sample_name, mIoU, acc = eva_on_one_sample(item)
        if constrain and constrain not in sample_name:
            continue
        if '_FLP' in sample_name:
            flip.append((mIoU, acc))
        elif '_R' in sample_name:
            rota.append((mIoU, acc))
        elif '_T' in sample_name:
            tran.append((mIoU, acc))
        elif '_S' in sample_name:
            scal.append((mIoU, acc))
        else:
            orig.append((mIoU, acc))
            
    for metrics, desc in [(orig, 'original'), (flip, 'flipped'), (rota, 'rotated'), (tran, 'translated'), (scal, 'rescaled')]:
        if not metrics:
            continue
        mIoUs, accs = 0, 0
        for mIoU, acc in metrics:
            mIoUs += mIoU
            accs += acc
        std_mIoU = np.std(np.array([i[0] for i in metrics]))
        std_acc = np.std(np.array([i[1] for i in metrics]))
        print(f'The average mIoU for {desc} is {mIoUs/len(metrics)} ± {std_mIoU}')
        print(f'The average accuracy for {desc} is {accs/len(metrics)} ± {std_acc}')
        

if __name__ == '__main__':
    mode = 'new'
    with_new_feature = True if mode == 'new' else False
    dataset_path = './dataset/test_set'
    output_path = f'./preds_{mode}.json'
    cfg_path = './config/default.yaml'
    checkpoint_path = f'./checkpoints/iMeshSegNet_mix_{mode}_17_Classes_32_f_best_DSC.ckpt'
    device = 'cuda'
    
    do_inf(dataset_path, cfg_path, checkpoint_path, output_path, with_new_features=with_new_feature)
    
    do_eva(output_path)
    # do_eva(output_path, constrain='lower')