import numpy as np
np.random.seed(1024)

import argparse
import os

from PIL import Image
from scipy.stats import truncnorm


parser = argparse.ArgumentParser()
parser.add_argument('src_dir', type=str)
parser.add_argument('dst_dir', type=str)
parser.add_argument('rect_dir', type=str)
parser.add_argument('-aug_size', type=int, default=5)
parser.add_argument('-extensions', type=str, nargs='+',
                    default=['png'])
args = parser.parse_args()


if not args.src_dir.endswith('/'):
    args.src_dir += '/'

if not args.dst_dir.endswith('/'):
    args.dst_dir += '/'

if not args.rect_dir.endswith('/'):
    args.rect_dir += '/'


def crop_by_small_area(img):
    img_array = np.asarray(img)
    ys, xs = (img_array < 200).nonzero()
    return img.crop((xs.min() - 1, ys.min() - 1, xs.max() + 2, ys.max() + 2))


def add_padding(img, l=0, r=0, t=0, b=0):
    img_array = np.asarray(img)
    shape = (img_array.shape[0] + t + b, img_array.shape[1] + l + r)
    padded_array = img_array.max() * np.ones(shape, dtype=img_array.dtype)
    padded_array[t:-b, l:-r] = img_array
    return Image.fromarray(padded_array)


def darker_paste(img, pasted, l, t):
    img_array = np.asarray(img).copy()
    pasted_array = np.asarray(pasted)
    b = t + pasted_array.shape[0]
    r = l + pasted_array.shape[1]
    img_array[t:b, l:r] = np.minimum(img_array[t:b, l:r], pasted_array)
    return Image.fromarray(img_array)


rectangles = []
for name in os.listdir(args.rect_dir):

    if name.split('.')[-1] not in args.extensions:
        continue

    img_rect = Image.open(args.rect_dir + name).convert('L')
    img_rect = crop_by_small_area(img_rect)
    rectangles.append(img_rect)


for name in os.listdir(args.src_dir):
    name_splits = name.split('.')
    if name_splits[-1] not in args.extensions:
        continue
    base_name = '.'.join(name_splits[:-1])

    for i in range(args.aug_size):
        img_num = Image.open(args.src_dir + name).convert('L')
        img_num = crop_by_small_area(img_num)

        num_size = np.random.randint(34, 48)
        if img_num.size[0] < img_num.size[1]:
            size = (int(img_num.size[0] * num_size / img_num.size[1]), num_size)
        else:
            size = (num_size, int(img_num.size[1] * num_size / img_num.size[0]))
        img_num = img_num.resize(size, Image.BICUBIC)

        img_rect = rectangles[np.random.randint(len(rectangles))]
        rect_size = np.random.randint(54, 64)
        size = (int(img_rect.size[0] * rect_size / img_rect.size[1]), rect_size)
        img_rect = img_rect.resize(size, Image.BICUBIC)
        img_rect = add_padding(img_rect, l=5, r=5, t=5, b=15)

        left = np.random.randint(0, img_rect.size[0] - img_num.size[0])
        top = np.random.randint(0, img_rect.size[1] - img_num.size[1])
        img_rect = darker_paste(img_rect, img_num, left, top)

        left = np.random.randint(left - (48 - img_num.size[0]), left + 1)
        top = np.random.randint(top - (48 - img_num.size[1]), top + 1)
        cropped = np.asarray(img_rect)[max(0, top):top + 48, max(0, left):left + 48]

        lined_array = 255 * np.ones((48, 48), dtype=np.uint8)
        left, top = max(0, -left), max(0, -top)
        lined_array[top:top + cropped.shape[0], left:left + cropped.shape[1]] = cropped
        img_lined = Image.fromarray(lined_array)

        img_lined.save(args.dst_dir + base_name + '_lined_' + str(i) + '.png')
