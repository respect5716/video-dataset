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
    

if __name__ == '__main__':
    main(args)