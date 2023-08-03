## structure of dataset directory
    -- dataset
        -- 3D_scans_per_patient_obj_files_b1
            -- OEJBIPTC
                -- OEJBIPTC_lower.obj
                -- OEJBIPTC_upper.obj
            -- ground-truth_labels_instances_b1
                -- OEJBIPTC
                    -- OEJBIPTC_lower.json
                    -- OEJBIPTC_upper.json
        -- 3D_scans_per_patient_obj_files_b2
        -- test_set
        -- test_set_stl

## for inference and visualize on single sample
`pip install requirements.txt`

`python test_inf.py`

## for evaluation on the test set
first `python make_test_set.py` to generate downsampled and cellwise labeled .vtk samples

then run `python evaluation.py` 

the `do_inf` function will predict on each sample, output with a json file

the `do_eva` function will read the predicted results from that json file and evaluate the mIoU and accuracy


## for loss curve review
`tensorboard --logdir=./checkpoints/mix_ori_epoch89/version_20`

or

`tensorboard --logdir=./checkpoints/mix_new_epoch339/version_19`


## Contribution
Contribute for the codes except for those under `./model`.


## Implement Details

This project is for evaluation only.

1. To start with, put the .ckpt checkpoint under `./checkpoints`, and the .yaml under `./config`.
2. Run `make_test_set.py` to generate the downsampled and labeled .vtk samples for evaluation. Must specify the test list path first (eg. `./tst_list.csv`, which should be splited before training), or use `read_filenames()` and `filelist_checker()` to read from the original dataset.
3. Run `evaluation.py` to evaluation on all the vtk samples under `./dataset/test_set`.
   1. `do_inf()` will load the model and do inference on samples, output a json named `preds_new.json` or `preds_old.json` which records both ground truth and predict result for every sample.
   2. `do_eva()` will read from the previous json file and evaluate on every sample, output a `evals.txt` records all the mIoU and accuarcy. The average mIoU and acc will be printed on the terminal, plz remember to copy and paste it to `evaluation_record.txt` manually.
4. Other functions in `evaluation.py`
   1. If wanting to test on augmentation items, call `do_augmentation()`.
   2. Call `make_augmentated_stl()` to output .stl form of samples for better visualization access. 
   3. Can also call `vtk2stl()` to convert all .vtk files under ip_dir into .stl and output to des_dir.
5. Run `test_inf.py` to do inference and visualization on selected sample.