import numpy as np
np.random.seed(1024)

import argparse
import os

from PIL import Image
from scipy.stats import truncnorm


parser = argparse.ArgumentParser()
parser.add_argument('src_dir', type=str)
parser.add_argument('dst_dir', type=str)
parser.add_argument('-aug_size', type=int, default=10)
parser.add_argument('-crop_range_low', type=float, default=0.4)
parser.add_argument('-crop_range_high', type=float, default=0.85)
parser.add_argument('-extensions', type=str, nargs='+',
                    default=['png', 'jpg'])
parser.add_argument('-use_normal', action='store_true')
args = parser.parse_args()

if not args.src_dir.endswith('/'):
    args.src_dir += '/'

if not args.dst_dir.endswith('/'):
    args.dst_dir += '/'

file_names = os.listdir(args.src_dir)
for name in file_names:
    name_splits = name.split('.')
    if name_splits[-1] not in args.extensions:
        continue
    base_name = '.'.join(name_splits[:-1])

    img = Image.open(args.src_dir + name)
    img_size = np.asarray(img.size)

    for i in range(args.aug_size):
        crop_ratio = np.random.uniform(args.crop_range_low, args.crop_range_high)
        crop_size = (img_size * crop_ratio).astype(int)

        left_top_region = img_size - crop_size
        if args.use_normal:
            mean, std = left_top_region / 2, left_top_region / 4

            # because truncnorm is designed over the domain of the standard normal,
            # we should convert min (a) and max (b) based on the Notes in bellow page.
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
            a, b = -mean / std, (left_top_region - mean) / std

            left_top = (int(truncnorm.rvs(a[0], b[0], loc=mean[0], scale=std[0])),
                        int(truncnorm.rvs(a[1], b[1], loc=mean[1], scale=std[1])))
        else:
            left_top = (np.random.randint(0, left_top_region[0]),
                        np.random.randint(0, left_top_region[1]))

        crop_img = img.crop(left_top + (left_top[0] + crop_size[0], left_top[1] + crop_size[1]))
        crop_img.save(args.dst_dir + base_name + '_augmented_' + str(i) + '.png')
