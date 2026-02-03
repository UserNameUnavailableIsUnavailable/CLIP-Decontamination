"""
python datasets/cvt_whu.py data/dataset_root -o data/converted_dataset
"""
import argparse
import os
import os.path as osp
import cv2

from mmengine.utils import mkdir_or_exist


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert dataset labels to mmsegmentation format')
    parser.add_argument('dataset_path', help='Dataset root folder path containing train/val/test splits')
    parser.add_argument('-o', '--out_dir', help='output path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    dataset_path = args.dataset_path
    if args.out_dir is None:
        out_dir = osp.join('data', 'converted_dataset')
    else:
        out_dir = args.out_dir

    splits = ['train', 'val', 'test']
    for split in splits:
        split_path = osp.join(dataset_path, split)
        if not osp.exists(split_path):
            print(f'Split {split} not found, skipping...')
            continue

        label_path = osp.join(split_path, 'OUT')
        if not osp.exists(label_path):
            print(f'OUT folder not found in {split}, skipping...')
            continue

        print(f'Processing {split}...')
        out_split_dir = osp.join(out_dir, split)
        mkdir_or_exist(out_split_dir)
        mkdir_or_exist(osp.join(out_split_dir, 'label_cvt'))

        for img_name in os.listdir(label_path):
            img = cv2.imread(osp.join(label_path, img_name), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f'Failed to read {img_name}, skipping...')
                continue
            img[img < 128] = 0
            img[img >= 128] = 1
            cv2.imwrite(osp.join(out_split_dir, 'label_cvt', img_name), img)

    print('Done!')


if __name__ == '__main__':
    main()