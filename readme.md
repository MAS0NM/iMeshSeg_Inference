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
