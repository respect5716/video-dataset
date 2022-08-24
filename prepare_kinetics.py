from torchvision.datasets import Kinetics

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=str, default='700')
parser.add_argument('--output_dir', type=str, default='kinetics')
parser.add_argument('--num_download_workers', type=int, default=10)
args = parser.parse_args()


def main(args):
    root = f'{args.output_dir}-{args.num_classes}'
    Kinetics(root=root, split='train', frames_per_clip=32, num_classes=args.num_classes, download=True, num_download_workers=args.num_download_workers)
    Kinetics(root=root, split='val', frames_per_clip=32, num_classes=args.num_classes, download=True, num_download_workers=args.num_download_workers)

    ### In case the process stopped
    """
    from tqdm import tqdm
    from torchvision.datasets.utils import download_and_extract_archive

    files = open('kinetics-700/files/k700_2020_train_path.txt', 'r').read().strip().split('\n')

    tars = sorted(glob('kinetics-700/tars/*_train_*.tar.gz'))
    tars = [i.split('/')[-1] for i in tars]

    remains = [f for f in files if f.split('/')[-1] not in tars]
    for line in tqdm(remains):
        name = line.split('/')[-1]
        tar_path = f'kinetics-700/tars/{name}'
        extract_root = 'kinetics-700/train/'
        download_and_extract_archive(line, tar_path, extract_root)
    """

if __name__ == '__main__':
    main(args)