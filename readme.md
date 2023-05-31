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

## to start
`pip install requirements.txt`

`python test_inf.py`