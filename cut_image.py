import numpy as np
import glob
import mmcv
import os.path as osp

from tqdm import tqdm
from PIL import Image, ImageOps


_INPUT_DIR = r'C:\Users\Kun\Desktop\compare_results'
_SIZE = 128
_EXT_NAME = '.png'


def find_bbox(image: Image):
    ivt_image = ImageOps.invert(image)
    # left, upper, right, and lower
    bbox = ivt_image.getbbox()
    return bbox


def main():
    files = glob.glob(osp.join(_INPUT_DIR, f'*{_EXT_NAME}'))

    left, upper, right, lower = find_bbox(Image.open(files[0]))
    margin = 0
    # add margin
    left = max(left - margin, 0)
    right = min(right + margin, _SIZE - 1)
    upper = max(upper - margin, 0)
    lower = min(lower + margin, _SIZE - 1)

    for fn in tqdm(files):
        result = np.full((_SIZE, _SIZE, 3), 255, dtype=np.uint8)
        image = mmcv.imread(fn).astype(np.uint8)
        # centered location
        left_ = (_SIZE - (right - left + 1)) // 2
        right_ = left_ + (right - left + 1)
        upper_ = (_SIZE - (lower - upper + 1)) // 2
        lower_ = upper_ + (lower - upper + 1)

        result[upper_: lower_, left_: right_, :] = image[upper: lower + 1, left: right + 1, :]
        mmcv.imwrite(result, osp.join(_INPUT_DIR, osp.basename(fn)[: -len(_EXT_NAME)] + '_out.png'))


if __name__ == '__main__':
    main()
