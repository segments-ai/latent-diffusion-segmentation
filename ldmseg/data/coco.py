"""
Author: Wouter Van Gansbeke

Dataset class for COCO Panoptic Segmentation
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import os
import json
import torch

import numpy as np
import torch.utils.data as data
from PIL import Image
from typing import Optional, Any, Tuple
import random
from collections import defaultdict

from ldmseg.data.util.mypath import MyPath
from ldmseg.utils.utils import color_map
from ldmseg.data.util.mask_generator import MaskingGenerator


class COCO(data.Dataset):
    COCO_CATEGORIES = [
        {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "person"},
        {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "bicycle"},
        {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "car"},
        {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "motorcycle"},
        {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": "airplane"},
        {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": "bus"},
        {"color": [0, 80, 100], "isthing": 1, "id": 7, "name": "train"},
        {"color": [0, 0, 70], "isthing": 1, "id": 8, "name": "truck"},
        {"color": [0, 0, 192], "isthing": 1, "id": 9, "name": "boat"},
        {"color": [250, 170, 30], "isthing": 1, "id": 10, "name": "traffic light"},
        {"color": [100, 170, 30], "isthing": 1, "id": 11, "name": "fire hydrant"},
        {"color": [220, 220, 0], "isthing": 1, "id": 13, "name": "stop sign"},
        {"color": [175, 116, 175], "isthing": 1, "id": 14, "name": "parking meter"},
        {"color": [250, 0, 30], "isthing": 1, "id": 15, "name": "bench"},
        {"color": [165, 42, 42], "isthing": 1, "id": 16, "name": "bird"},
        {"color": [255, 77, 255], "isthing": 1, "id": 17, "name": "cat"},
        {"color": [0, 226, 252], "isthing": 1, "id": 18, "name": "dog"},
        {"color": [182, 182, 255], "isthing": 1, "id": 19, "name": "horse"},
        {"color": [0, 82, 0], "isthing": 1, "id": 20, "name": "sheep"},
        {"color": [120, 166, 157], "isthing": 1, "id": 21, "name": "cow"},
        {"color": [110, 76, 0], "isthing": 1, "id": 22, "name": "elephant"},
        {"color": [174, 57, 255], "isthing": 1, "id": 23, "name": "bear"},
        {"color": [199, 100, 0], "isthing": 1, "id": 24, "name": "zebra"},
        {"color": [72, 0, 118], "isthing": 1, "id": 25, "name": "giraffe"},
        {"color": [255, 179, 240], "isthing": 1, "id": 27, "name": "backpack"},
        {"color": [0, 125, 92], "isthing": 1, "id": 28, "name": "umbrella"},
        {"color": [209, 0, 151], "isthing": 1, "id": 31, "name": "handbag"},
        {"color": [188, 208, 182], "isthing": 1, "id": 32, "name": "tie"},
        {"color": [0, 220, 176], "isthing": 1, "id": 33, "name": "suitcase"},
        {"color": [255, 99, 164], "isthing": 1, "id": 34, "name": "frisbee"},
        {"color": [92, 0, 73], "isthing": 1, "id": 35, "name": "skis"},
        {"color": [133, 129, 255], "isthing": 1, "id": 36, "name": "snowboard"},
        {"color": [78, 180, 255], "isthing": 1, "id": 37, "name": "sports ball"},
        {"color": [0, 228, 0], "isthing": 1, "id": 38, "name": "kite"},
        {"color": [174, 255, 243], "isthing": 1, "id": 39, "name": "baseball bat"},
        {"color": [45, 89, 255], "isthing": 1, "id": 40, "name": "baseball glove"},
        {"color": [134, 134, 103], "isthing": 1, "id": 41, "name": "skateboard"},
        {"color": [145, 148, 174], "isthing": 1, "id": 42, "name": "surfboard"},
        {"color": [255, 208, 186], "isthing": 1, "id": 43, "name": "tennis racket"},
        {"color": [197, 226, 255], "isthing": 1, "id": 44, "name": "bottle"},
        {"color": [171, 134, 1], "isthing": 1, "id": 46, "name": "wine glass"},
        {"color": [109, 63, 54], "isthing": 1, "id": 47, "name": "cup"},
        {"color": [207, 138, 255], "isthing": 1, "id": 48, "name": "fork"},
        {"color": [151, 0, 95], "isthing": 1, "id": 49, "name": "knife"},
        {"color": [9, 80, 61], "isthing": 1, "id": 50, "name": "spoon"},
        {"color": [84, 105, 51], "isthing": 1, "id": 51, "name": "bowl"},
        {"color": [74, 65, 105], "isthing": 1, "id": 52, "name": "banana"},
        {"color": [166, 196, 102], "isthing": 1, "id": 53, "name": "apple"},
        {"color": [208, 195, 210], "isthing": 1, "id": 54, "name": "sandwich"},
        {"color": [255, 109, 65], "isthing": 1, "id": 55, "name": "orange"},
        {"color": [0, 143, 149], "isthing": 1, "id": 56, "name": "broccoli"},
        {"color": [179, 0, 194], "isthing": 1, "id": 57, "name": "carrot"},
        {"color": [209, 99, 106], "isthing": 1, "id": 58, "name": "hot dog"},
        {"color": [5, 121, 0], "isthing": 1, "id": 59, "name": "pizza"},
        {"color": [227, 255, 205], "isthing": 1, "id": 60, "name": "donut"},
        {"color": [147, 186, 208], "isthing": 1, "id": 61, "name": "cake"},
        {"color": [153, 69, 1], "isthing": 1, "id": 62, "name": "chair"},
        {"color": [3, 95, 161], "isthing": 1, "id": 63, "name": "couch"},
        {"color": [163, 255, 0], "isthing": 1, "id": 64, "name": "potted plant"},
        {"color": [119, 0, 170], "isthing": 1, "id": 65, "name": "bed"},
        {"color": [0, 182, 199], "isthing": 1, "id": 67, "name": "dining table"},
        {"color": [0, 165, 120], "isthing": 1, "id": 70, "name": "toilet"},
        {"color": [183, 130, 88], "isthing": 1, "id": 72, "name": "tv"},
        {"color": [95, 32, 0], "isthing": 1, "id": 73, "name": "laptop"},
        {"color": [130, 114, 135], "isthing": 1, "id": 74, "name": "mouse"},
        {"color": [110, 129, 133], "isthing": 1, "id": 75, "name": "remote"},
        {"color": [166, 74, 118], "isthing": 1, "id": 76, "name": "keyboard"},
        {"color": [219, 142, 185], "isthing": 1, "id": 77, "name": "cell phone"},
        {"color": [79, 210, 114], "isthing": 1, "id": 78, "name": "microwave"},
        {"color": [178, 90, 62], "isthing": 1, "id": 79, "name": "oven"},
        {"color": [65, 70, 15], "isthing": 1, "id": 80, "name": "toaster"},
        {"color": [127, 167, 115], "isthing": 1, "id": 81, "name": "sink"},
        {"color": [59, 105, 106], "isthing": 1, "id": 82, "name": "refrigerator"},
        {"color": [142, 108, 45], "isthing": 1, "id": 84, "name": "book"},
        {"color": [196, 172, 0], "isthing": 1, "id": 85, "name": "clock"},
        {"color": [95, 54, 80], "isthing": 1, "id": 86, "name": "vase"},
        {"color": [128, 76, 255], "isthing": 1, "id": 87, "name": "scissors"},
        {"color": [201, 57, 1], "isthing": 1, "id": 88, "name": "teddy bear"},
        {"color": [246, 0, 122], "isthing": 1, "id": 89, "name": "hair drier"},
        {"color": [191, 162, 208], "isthing": 1, "id": 90, "name": "toothbrush"},
        {"color": [255, 255, 128], "isthing": 0, "id": 92, "name": "banner"},
        {"color": [147, 211, 203], "isthing": 0, "id": 93, "name": "blanket"},
        {"color": [150, 100, 100], "isthing": 0, "id": 95, "name": "bridge"},
        {"color": [168, 171, 172], "isthing": 0, "id": 100, "name": "cardboard"},
        {"color": [146, 112, 198], "isthing": 0, "id": 107, "name": "counter"},
        {"color": [210, 170, 100], "isthing": 0, "id": 109, "name": "curtain"},
        {"color": [92, 136, 89], "isthing": 0, "id": 112, "name": "door-stuff"},
        {"color": [218, 88, 184], "isthing": 0, "id": 118, "name": "floor-wood"},
        {"color": [241, 129, 0], "isthing": 0, "id": 119, "name": "flower"},
        {"color": [217, 17, 255], "isthing": 0, "id": 122, "name": "fruit"},
        {"color": [124, 74, 181], "isthing": 0, "id": 125, "name": "gravel"},
        {"color": [70, 70, 70], "isthing": 0, "id": 128, "name": "house"},
        {"color": [255, 228, 255], "isthing": 0, "id": 130, "name": "light"},
        {"color": [154, 208, 0], "isthing": 0, "id": 133, "name": "mirror-stuff"},
        {"color": [193, 0, 92], "isthing": 0, "id": 138, "name": "net"},
        {"color": [76, 91, 113], "isthing": 0, "id": 141, "name": "pillow"},
        {"color": [255, 180, 195], "isthing": 0, "id": 144, "name": "platform"},
        {"color": [106, 154, 176], "isthing": 0, "id": 145, "name": "playingfield"},
        {"color": [230, 150, 140], "isthing": 0, "id": 147, "name": "railroad"},
        {"color": [60, 143, 255], "isthing": 0, "id": 148, "name": "river"},
        {"color": [128, 64, 128], "isthing": 0, "id": 149, "name": "road"},
        {"color": [92, 82, 55], "isthing": 0, "id": 151, "name": "roof"},
        {"color": [254, 212, 124], "isthing": 0, "id": 154, "name": "sand"},
        {"color": [73, 77, 174], "isthing": 0, "id": 155, "name": "sea"},
        {"color": [255, 160, 98], "isthing": 0, "id": 156, "name": "shelf"},
        {"color": [255, 255, 255], "isthing": 0, "id": 159, "name": "snow"},
        {"color": [104, 84, 109], "isthing": 0, "id": 161, "name": "stairs"},
        {"color": [169, 164, 131], "isthing": 0, "id": 166, "name": "tent"},
        {"color": [225, 199, 255], "isthing": 0, "id": 168, "name": "towel"},
        {"color": [137, 54, 74], "isthing": 0, "id": 171, "name": "wall-brick"},
        {"color": [135, 158, 223], "isthing": 0, "id": 175, "name": "wall-stone"},
        {"color": [7, 246, 231], "isthing": 0, "id": 176, "name": "wall-tile"},
        {"color": [107, 255, 200], "isthing": 0, "id": 177, "name": "wall-wood"},
        {"color": [58, 41, 149], "isthing": 0, "id": 178, "name": "water-other"},
        {"color": [183, 121, 142], "isthing": 0, "id": 180, "name": "window-blind"},
        {"color": [255, 73, 97], "isthing": 0, "id": 181, "name": "window-other"},
        {"color": [107, 142, 35], "isthing": 0, "id": 184, "name": "tree-merged"},
        {"color": [190, 153, 153], "isthing": 0, "id": 185, "name": "fence-merged"},
        {"color": [146, 139, 141], "isthing": 0, "id": 186, "name": "ceiling-merged"},
        {"color": [70, 130, 180], "isthing": 0, "id": 187, "name": "sky-other-merged"},
        {"color": [134, 199, 156], "isthing": 0, "id": 188, "name": "cabinet-merged"},
        {"color": [209, 226, 140], "isthing": 0, "id": 189, "name": "table-merged"},
        {"color": [96, 36, 108], "isthing": 0, "id": 190, "name": "floor-other-merged"},
        {"color": [96, 96, 96], "isthing": 0, "id": 191, "name": "pavement-merged"},
        {"color": [64, 170, 64], "isthing": 0, "id": 192, "name": "mountain-merged"},
        {"color": [152, 251, 152], "isthing": 0, "id": 193, "name": "grass-merged"},
        {"color": [208, 229, 228], "isthing": 0, "id": 194, "name": "dirt-merged"},
        {"color": [206, 186, 171], "isthing": 0, "id": 195, "name": "paper-merged"},
        {"color": [152, 161, 64], "isthing": 0, "id": 196, "name": "food-other-merged"},
        {"color": [116, 112, 0], "isthing": 0, "id": 197, "name": "building-other-merged"},
        {"color": [0, 114, 143], "isthing": 0, "id": 198, "name": "rock-merged"},
        {"color": [102, 102, 156], "isthing": 0, "id": 199, "name": "wall-other-merged"},
        {"color": [250, 141, 255], "isthing": 0, "id": 200, "name": "rug-merged"},
    ]

    COCO_CATEGORY_NAMES = [k["name"] for k in COCO_CATEGORIES]

    def __init__(
        self,
        prefix: str,
        split: str = 'val',
        tokenizer: Optional[Any] = None,
        transform: Optional[Any] = None,
        download: bool = False,
        remap_labels: bool = False,
        caption_dropout: float = 0.0,
        overfit: bool = False,
        encoding_mode: str = 'color',
        caption_type: str = 'none',
        inpaint_mask_size: Optional[Tuple[int]] = None,
        num_classes: int = 128,
        fill_value: int = 0.5,
        ignore_label: int = 0,
        inpainting_strength: float = 0.0,
    ):
        """
        Args:
            prefix (str): path to dataset
            split (str): train, val or test
            tokenizer (Optional[Any]): tokenizer
            transform (Optional[Any]): transform
            download (bool): download dataset
            remap_labels (bool): remap labels to random id from 0 to N
            caption_dropout (float): probability to drop caption
            overfit (bool): overfit to a small set of images
            encoding_mode (str): color, random_color, bits, none
            caption_type (str): none, class_label
            inpaint_mask_size (Optional[Tuple[int]]): inpainting mask size which will be used for inpainting
        """

        # Set paths
        root = MyPath.db_root_dir('coco', prefix=prefix)
        self.root = root
        self.prefix = prefix
        valid_splits = ['train', 'val', 'test']
        print(split)
        assert (split in valid_splits)
        assert not download
        self.split = split
        self.tokenizer = tokenizer
        self.caption_dropout = caption_dropout

        self.num_classes = num_classes    # max 128 intances in panoptic segmentation
        self.ignore_label = ignore_label  # void label for panoptic segmentation
        self.fill_value = fill_value      # fill value for bit encoding in case of void
        self.inpainting_strength = inpainting_strength   # inpainting strength (what percentage is already given?)
        self.remap_labels = remap_labels
        if inpaint_mask_size is None:
            inpaint_mask_size = (64, 64)
        self.maskgenerator = MaskingGenerator(input_size=inpaint_mask_size, mode='random_local')

        # Transform
        self.transform = transform
        self.cmap = color_map()

        # Splits are pre-cut
        print("Initializing dataloader for COCO {} set".format(''.join(self.split)))
        if self.split == 'train':
            file_dir = 'train2017'
            self.training = True
        elif self.split == 'val':
            file_dir = 'val2017'
            self.training = False
        else:
            raise NotImplementedError

        # define paths
        _image_dir = os.path.join(self.root, file_dir)
        _semseg_dir = os.path.join(self.root, f"annotations/panoptic_{file_dir}")
        lines = os.listdir(os.path.join(self.root, file_dir))
        lines = sorted([line.split('.')[0] for line in lines])

        panoptic_json = os.path.join(self.root, f"annotations/panoptic_{file_dir}.json")
        captions_json = os.path.join(self.root, f"annotations/captions_{file_dir}.json")
        blip_captions_json = os.path.join('data/blip_captions', f"captions_{file_dir}.json")
        if not os.path.isfile(blip_captions_json):
            blip_captions_json = None
        self.panoptic_json = panoptic_json
        self.panoptic_root = _semseg_dir

        # load annotations
        with open(panoptic_json, "r") as f:
            self.panoptic_anns = json.load(f)
        with open(captions_json, "r") as f:
            self.captions_anns = json.load(f)
        self.blip_captions_anns = blip_captions_json
        if self.blip_captions_anns is not None:
            with open(blip_captions_json, "r") as f:
                self.blip_captions_anns = json.load(f)
        self.captions_dict = defaultdict(list)
        for ann in self.captions_anns['annotations']:
            self.captions_dict[ann['image_id']].append(ann['caption'])
        self.annotations_dict = {ann['file_name']: ann for ann in self.panoptic_anns['annotations']}
        categories = self.panoptic_anns["categories"]
        cat_info = {cat['id']: {'name': cat['name'], 'isthing': cat['isthing']} for cat in categories}
        self.categories = categories
        self.cat_info = cat_info

        # meta data for remapping labels
        self.meta_data = self.get_metadata()

        self.images = []
        self.semsegs = []

        for ii, line in enumerate(lines):
            # Images dir
            _image = os.path.join(_image_dir, line + ".jpg")
            assert os.path.isfile(_image)
            self.images.append(_image)

            # Segmentation dir
            _semseg = os.path.join(_semseg_dir, line + '.png')
            assert os.path.isfile(_semseg)
            self.semsegs.append(_semseg)

        # filter out data without annotations
        images = []
        semsegs = []
        for image_path, semseg_path in zip(self.images, self.semsegs):
            base_name = os.path.basename(semseg_path)
            seg_info = self.annotations_dict[base_name]['segments_info']
            if len(seg_info) > 0 and not all([x['iscrowd'] == 1 for x in seg_info]):
                images.append(image_path)
                semsegs.append(semseg_path)
        print('filtered out {} images without annotations'.format(len(self.semsegs) - len(semsegs)))
        self.images = images
        self.semsegs = semsegs
        assert (len(self.images) == len(self.semsegs))

        # Uncomment to overfit to one image
        if overfit:
            n_of = 1000
            self.images = self.images[:n_of]
            self.semsegs = self.semsegs[:n_of]

        assert caption_type in ['none', 'caption', 'class_label', 'blip']
        assert encoding_mode in ['color', 'random_color', 'bits', 'none']
        self.encoding_mode = encoding_mode
        self.caption_type = caption_type
        print('caption type: {}'.format(self.caption_type))
        print('caption dropout: {}'.format(self.caption_dropout))
        print('encoding mode: {}'.format(self.encoding_mode))
        print('fill value: {}'.format(self.fill_value))
        print('inpainting strength: {}'.format(self.inpainting_strength))
        self.background_to_ignore = False
        self.ignore_to_background = False
        if self.training:
            self.pixel_threshold = 10
        else:
            self.pixel_threshold = 0

        # Display stats
        print('Number of dataset images: {:d}'.format(len(self.images)))

    def _remap_labels_fn(self, labels, max_val=None, keep_background_fixed=True):
        # keep the original background class index
        # max val is the maximum number of classes to remap to
        # ignore index is kept fixed

        # remapping only works if additional background classes are ordered from 0 to N.
        max_val = max_val if max_val is not None else self.num_classes
        unique_values = [x for x in np.unique(labels) if x != self.ignore_label]
        assert len(unique_values) < max_val, f"Number of unique values {len(unique_values)} is larger or equal than max_val {max_val}"  # noqa

        # np.random.seed(1)
        targets = np.random.choice(max_val - 1,
                                   size=len(unique_values),
                                   replace=False)  # sampling without replacement
        targets = targets + 1

        # mapping dict
        mapping = dict(zip(unique_values, targets))
        remapped_labels = np.full(labels.shape, self.ignore_label, dtype=labels.dtype)
        for val, remap_val in mapping.items():
            remapped_labels[labels == val] = remap_val
        mapping_np = np.full(self.num_classes, -1, dtype=int)
        for idx, (_, remap_val) in enumerate(mapping.items()):
            mapping_np[idx] = remap_val

        # sanity checks: make sure all target values are smaller than max_val and unique
        assert np.all(mapping_np[mapping_np != -1] < max_val)
        assert np.all(mapping_np[mapping_np != -1] >= 0)
        assert len(np.unique(mapping_np[mapping_np != -1])) == len(mapping_np[mapping_np != -1])
        assert len(np.unique(mapping_np[mapping_np != -1])) == len(unique_values)

        return remapped_labels, mapping

    def encode_semseg(self, semseg, cmap=None):
        # we will encode the semseg map with a color map
        if cmap is None:
            cmap = color_map()
        seg_t = semseg.astype(np.uint8)
        array_seg_t = np.full((seg_t.shape[0], seg_t.shape[1], cmap.shape[1]), self.ignore_label, dtype=cmap.dtype)
        for class_i in np.unique(seg_t):
            array_seg_t[seg_t == class_i] = cmap[class_i]
        return array_seg_t

    def encode_semseg_random(self, semseg, cmap=None):
        seg_t = semseg.astype(np.uint8)
        color_palette = set()
        array_seg_t = np.full((seg_t.shape[0], seg_t.shape[1], cmap.shape[1]), self.ignore_label, dtype=cmap.dtype)
        unique_classes = np.unique(seg_t)
        while len(color_palette) < len(unique_classes):
            color_palette.add(tuple(np.random.choice(range(256), size=3)))
        for class_i in unique_classes:
            if class_i == self.ignore_label:
                continue
            array_seg_t[seg_t == class_i] = color_palette.pop()
        assert array_seg_t.max() < 256
        return array_seg_t

    def encode_bitmap(self, x: torch.Tensor, n: int = 7, fill_value: float = 0.5):
        ignore_mask = x == self.ignore_label
        x = torch.bitwise_right_shift(x, torch.arange(n, device=x.device)[:, None, None])  # shift with n bits
        x = torch.remainder(x, 2).float()                                                  # take modulo 2 to get 0 or 1
        x[:, ignore_mask] = fill_value                                                     # set invalid pixels to 0.5
        return x, ignore_mask

    def decode_bitmap(self, x: torch.Tensor, n: int = 7):
        x = (x > 0.).float()                                          # output between -1 and 1
        n = x.shape[0]                                                # number of channels = number of bits
        x = x * 2 ** torch.arange(n, device=x.device)[:, None, None]  # get the value of each bit
        x = torch.sum(x, dim=0)                                       # sum over bits (no keepdim!)
        x = x.long()                                                  # cast to int64 (or long)
        return x

    def get_inpainting_mask(self, strength=0.5):
        mask = self.maskgenerator(t=strength)
        mask = torch.from_numpy(mask).bool()
        return mask

    def __getitem__(self, index):
        sample = {}
        mapping = dict()

        # Load image
        _img = self._load_img(index, mode='pil')
        sample['image'] = _img

        # Load pixel-level annotations
        _semseg, segments_info, captions_info, key_id = self._load_semseg(index, mode='array')
        unique_classes = np.unique(_semseg)
        unique_classes = unique_classes[unique_classes != self.ignore_label]

        # handle caption
        if self.caption_type == 'caption':
            sample['text'] = random.choice(captions_info) if self.training else captions_info[0]
        elif self.caption_type == 'class_label':
            category_names = [v['category_name'] for v in segments_info.values()]
            sample['text'] = ', '.join(category_names)
        elif self.caption_type == 'blip' and self.blip_captions_anns is not None:
            sample['text'] = self.blip_captions_anns[key_id]
        else:
            sample['text'] = ""

        # dropout caption
        if self.training and self.caption_dropout > random.random():
            sample['text'] = ""

        # remap labels
        if self.remap_labels:
            _semseg, mapping = self._remap_labels_fn(_semseg, max_val=self.num_classes, keep_background_fixed=True)
            segments_info = {mapping[key]: val for key, val in segments_info.items()}
            assert len(unique_classes) == len(segments_info)

        assert _semseg.max() < 256
        sample['semseg'] = _semseg.astype(np.uint8)
        sample['semseg'] = Image.fromarray(sample['semseg'])

        # mask with ones for valid pixels
        sample['mask'] = np.ones_like(_semseg)
        sample['mask'] = Image.fromarray(sample['mask'])

        # encode semseg
        if self.encoding_mode == 'random_color':
            sample['image_semseg'] = self.encode_semseg_random(_semseg, cmap=self.cmap)
            sample['image_semseg'] = Image.fromarray(sample['image_semseg'])
        elif self.encoding_mode == 'color':
            sample['image_semseg'] = self.encode_semseg(_semseg, cmap=self.cmap)
            sample['image_semseg'] = Image.fromarray(sample['image_semseg'])

        # meta data
        sample['meta'] = {
            'im_size': (_img.size[1], _img.size[0]),
            'image_file': self.images[index],
            "image_id": int(os.path.basename(self.images[index]).split(".")[0]),
            'segments_info': segments_info,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        # after transforms
        if self.encoding_mode == 'bits':
            sample['image_semseg'], _ = self.encode_bitmap(sample['semseg'], n=7, fill_value=self.fill_value)
        elif self.encoding_mode == 'none':
            sample['image_semseg'] = sample['semseg'].unsqueeze(0).repeat(3, 1, 1).float() / self.num_classes

        # get tokens
        if self.tokenizer is not None:
            sample['tokens'] = self.tokenizer(sample['text'],
                                              padding='max_length',
                                              max_length=77,
                                              truncation=True,
                                              return_tensors='pt').input_ids.squeeze(0)

        sample['inpainting_mask'] = self.get_inpainting_mask(strength=self.inpainting_strength)

        return sample

    def get_class_names(self):
        return self.COCO_CATEGORY_NAMES

    def __len__(self):
        return len(self.images)

    def _load_img(self, index, mode='array'):
        _img = Image.open(self.images[index]).convert('RGB')
        if mode == 'pil':
            return _img
        return np.array(_img)

    def _load_semseg(self, index, mode='array'):
        _semseg = np.array(Image.open(self.semsegs[index]).convert('RGB'))
        _semseg = _semseg[:, :, 0] + 256 * _semseg[:, :, 1] + (256 ** 2) * _semseg[:, :, 2]

        # count pixels for each unique instance and set to ignore if below pixel threshold
        small_instances = set()
        if self.training:
            if self.pixel_threshold > 0:
                ids, counts = np.unique(_semseg, return_counts=True)
                for i, c in zip(ids, counts):
                    if c < self.pixel_threshold:
                        _semseg[_semseg == i] = self.ignore_label
                        small_instances.add(i)

        # load segments info
        key = self.semsegs[index].split('/')[-1]
        segments_info = self.annotations_dict[key]['segments_info']
        keep_segments_info = {}
        for seg in segments_info:
            if seg['id'] in small_instances:
                continue
            if seg['iscrowd'] and self.training:
                _semseg[_semseg == seg['id']] = self.ignore_label
                continue

            assert seg['iscrowd'] == 0 or not self.training
            keep_segments_info[seg['id']] = {'category_id': seg['category_id'],
                                             'iscrowd': seg['iscrowd'],
                                             'category_name': self.cat_info[seg['category_id']]['name'],
                                             'isthing': self.cat_info[seg['category_id']]['isthing']
                                             }
            # remap category ids to contiguous ids
            curr_cat_id = keep_segments_info[seg['id']]['category_id']
            if curr_cat_id in self.meta_data["thing_dataset_id_to_contiguous_id"]:
                keep_segments_info[seg['id']]['category_id'] = self.meta_data["thing_dataset_id_to_contiguous_id"][curr_cat_id]  # noqa
            else:
                keep_segments_info[seg['id']]['category_id'] = self.meta_data["stuff_dataset_id_to_contiguous_id"][curr_cat_id]  # noqa
            assert keep_segments_info[seg['id']]['category_id'] < 133  # 133 is the number of classes in COCO panoptic

        # load captions
        image_id = key.split('.')[0]
        captions_info = self.captions_dict[int(image_id)]

        # assert
        assert _semseg.max() > 0
        assert len(keep_segments_info) == len([x for x in np.unique(_semseg) if x != self.ignore_label])

        if mode == 'pil':
            return Image.fromarray(_semseg.astype(np.uint8))

        return _semseg, keep_segments_info, captions_info, image_id + '.jpg'

    def get_metadata(self):
        meta = {}
        # The following metadata maps contiguous id from [0, #thing categories +
        # #stuff categories) to their names and colors. We have to replica of the
        # same name and color under "thing_*" and "stuff_*" because the current
        # visualization function in D2 handles thing and class classes differently
        # due to some heuristic used in Panoptic FPN. We keep the same naming to
        # enable reusing existing visualization functions.
        thing_classes = [k["name"] for k in self.COCO_CATEGORIES if k["isthing"] == 1]
        thing_colors = [k["color"] for k in self.COCO_CATEGORIES if k["isthing"] == 1]
        stuff_classes = [k["name"] for k in self.COCO_CATEGORIES]
        stuff_colors = [k["color"] for k in self.COCO_CATEGORIES]

        meta["thing_classes"] = thing_classes
        meta["thing_colors"] = thing_colors
        meta["stuff_classes"] = stuff_classes
        meta["stuff_colors"] = stuff_colors

        # Convert category id for training:
        #   category id: like semantic segmentation, it is the class id for each
        #   pixel. Since there are some classes not used in evaluation, the category
        #   id is not always contiguous and thus we have two set of category ids:
        #       - original category id: category id in the original dataset, mainly
        #           used for evaluation.
        #       - contiguous category id: [0, #classes), in order to train the linear
        #           softmax classifier.
        thing_dataset_id_to_contiguous_id = {}
        stuff_dataset_id_to_contiguous_id = {}
        cat2name = {}

        for i, cat in enumerate(self.COCO_CATEGORIES):
            if cat["isthing"]:
                thing_dataset_id_to_contiguous_id[cat["id"]] = i
            # else:
            #     stuff_dataset_id_to_contiguous_id[cat["id"]] = i

            # in order to use sem_seg evaluator
            stuff_dataset_id_to_contiguous_id[cat["id"]] = i

            cat2name[cat['id']] = cat['name']

        meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
        meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id
        meta["cat2name"] = cat2name

        meta["panoptic_json"] = self.panoptic_json
        meta["panoptic_root"] = self.panoptic_root

        return meta

    def __str__(self):
        return 'COCO(split=' + str(self.split) + ')'

    def _validate_annotations_simple(self):
        from tqdm import tqdm

        for i in tqdm(range(len(self))):
            semseg, segment_info, _, _ = self._load_semseg(i)
            unique_labels = np.unique(semseg, return_counts=False)
            unique_labels = unique_labels[unique_labels != self.ignore_label]
            assert len(segment_info) == len(unique_labels)
            assert sorted(unique_labels) == sorted(segment_info.keys())
        return


if __name__ == '__main__':
    """ For purpose of debugging """
    import util.pil_transforms as pil_tr
    import torchvision.transforms as T

    size = 128
    transforms = T.Compose([
        pil_tr.RandomHorizontalFlip(),
        pil_tr.CropResize((size, size), crop_mode=None),
        pil_tr.ToTensor(),
    ])

    dataset = COCO(
        prefix='/home/ubuntu/datasets',
        split='train',
        transform=transforms,
        remap_labels=True,
    )
    res = dataset._validate_annotations_simple()
