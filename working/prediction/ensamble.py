import os
import pandas as pd
import json
from datetime import datetime



def get_results(result_file_path):
    with open(result_file_path, "r") as f:
        res = float(f.read().strip())
    return res

if __name__ == "__main__":
    folders = [
        "/mnt/pfs-mc0p4k/cv/team/panxuhao/playground/miccai24_cxr_lt/code/prediction/submission/swinv2-large-ft-focalloss-unbalanced-data/epoch_12",
        # "/mnt/pfs-mc0p4k/cv/team/panxuhao/playground/miccai24_cxr_lt/code/prediction/submission/mae_vit-large-mae-ft-unbalanced-data/epoch_36",
        "/mnt/pfs-mc0p4k/cv/team/panxuhao/playground/miccai24_cxr_lt/code/prediction/submission/swinv2-large-ft-focalloss-unbalanced-data_alltrain_data/epoch_12",
        "/mnt/pfs-mc0p4k/cv/team/panxuhao/playground/miccai24_cxr_lt/code/prediction/submission/swinv2-large-ft-focalloss-unbalanced-data_alltrain_data_e6/epoch_6",
        # "/mnt/pfs-mc0p4k/cv/team/panxuhao/playground/miccai24_cxr_lt/code/prediction/submission/swinv2-large-ft-focalloss-unbalanced-data_alltrain_data_e3/epoch_3",
        # "/mnt/pfs-mc0p4k/cv/team/panxuhao/playground/miccai24_cxr_lt/code/prediction/submission/swinv2-large-ft-focalloss-unbalanced-data_alltrain_data_e6/epoch_5",
        "/mnt/pfs-mc0p4k/cv/team/panxuhao/playground/miccai24_cxr_lt/code/prediction/submission/swinv2-large-ft-focalloss-unbalanced-data_alltrain_data_e12/epoch_12",
        "/mnt/pfs-mc0p4k/cv/team/panxuhao/playground/miccai24_cxr_lt/code/prediction/submission/swinv2-large-ft-focalloss-unbalanced-data_alltrain_data_e12/epoch_9",
    ]
    results_file_name = "results.txt"
    csv_name = "development_task1_sample_submission.csv"
    all_dfs = []
    all_res = []
    for f in folders:
        result_file_path = os.path.join(f, results_file_name)
        res = get_results(result_file_path)
        df = pd.read_csv(os.path.join(f, csv_name))
        all_dfs.append(df)
        all_res.append(res)
        print(f, res)
    
    des_folder = "/mnt/pfs-mc0p4k/cv/team/panxuhao/playground/miccai24_cxr_lt/code/prediction/submission_ensamble"
    now = datetime.now()
    timestamp = now.strftime('%Y-%m-%d_%H-%M-%S')
    output_folder = os.path.join(des_folder, timestamp)
    os.makedirs(output_folder, exist_ok=True)

    with open(os.path.join(output_folder, "raw_results.txt"), "w") as file:
        for f, r in zip(folders, all_res):
            file.write(f"{f}: {r}\n")
            
    total_res = sum(all_res)
    all_res = [res / total_res for res in all_res]

    dicom_id = all_dfs[0]['dicom_id']
    numeric_columns = all_dfs[0].columns[1:]

    weighted_sum = pd.DataFrame(0, index=all_dfs[0].index, columns=numeric_columns)

    for df, weight in zip(all_dfs, all_res):
        weighted_sum += df[numeric_columns] * weight
    result = pd.concat([dicom_id, weighted_sum], axis=1)
    
    result.to_csv(os.path.join(output_folder, csv_name), index=False)
    print(f"Output folder: {output_folder}")