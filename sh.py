import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
        "--data_path",
        required=True,
        type=str,
        help="Root directory for the dataset",
    )
parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU id to use",
    )
opt = parser.parse_args()
root_dir = opt.data_path

dirs1 = os.listdir(root_dir)
for dir1 in dirs1:
    if os.path.isdir(os.path.join(root_dir, dir1)):
        dirs = os.listdir(os.path.join(root_dir, dir1, 'test'))
        for dir2 in dirs:
            if dir2 != 'good':
               
                mask_dir = f'./data/ymliu/benchmark/anomalydiffusion/data/{dir1}/ground_truth/{dir2}/'
                
                os.system(
                    'CUDA_VISIBLE_DEVICES=%d python generate_with_mask.py --data_root=%s --sample_name=%s --anomaly_name=%s --mask_path=%s' % (
                    opt.gpu_id, opt.data_path, dir1, dir2, mask_dir))
