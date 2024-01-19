import random
import math
import numpy as np


class MaskingGenerator:
    def __init__(
            self,
            input_size=(32, 32),
            num_masking_patches=512,
            min_num_patches=4,
            max_num_patches=128,
            min_aspect=0.3,
            max_aspect=None,
            mode='random_global',
    ):

        if not isinstance(input_size, (tuple, list)):
            input_size = (input_size, ) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
        self.fill_percentage = float(self.num_masking_patches) / self.num_patches
        self.mode = mode

    def __repr__(self):
        repr_str = "Generator in mode %s with params (%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.mode,
            self.height, self.width, self.min_num_patches, self.max_num_patches,
            self.num_masking_patches, self.log_aspect_ratio[0], self.log_aspect_ratio[1])
        return repr_str

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for _ in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top: top + h, left: left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    def _get_global_mask(self, mask, verbose=False):
        mask_count = 0
        # self.num_masking_patches = random.randint(128, 1024)

        num_iters = 0
        while mask_count < self.num_masking_patches:
            max_mask_patches = self.num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)
            num_iters += 1

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        if verbose:
            print('num_iters =', num_iters)
        return mask

    def _get_local_mask(self, mask, verbose=False, strength=0.5):
        mask[np.random.rand(*self.get_shape()) < strength] = 1
        if verbose:
            print('mask.sum() =', mask.sum())
        return mask

    def __call__(self, t=0.5, verbose=False):
        # init mask with zeros
        mask = np.zeros(shape=self.get_shape(), dtype=np.int64)

        if self.mode == 'random_local':
            return self._get_local_mask(mask, verbose=verbose, strength=t)

        elif self.mode == 'random_global':
            return self._get_global_mask(mask, verbose=verbose)

        elif self.mode == 'random_global_plus_local':
            return (self._get_global_mask(mask, verbose=verbose) +
                    self._get_local_mask(mask, verbose=verbose, strength=t)) > 0

        elif self.mode == 'object':
            assert False

        else:
            raise NotImplementedError


if __name__ == "__main__":

    from PIL import Image

    # generator = MaskingGenerator((32, 32), 512, 16, 32, mode='random_global_plus_local')
    generator = MaskingGenerator((64, 64), mode='random_local')
    print(generator)
    mask = generator(verbose=True, t=0.15)

    # save mask as image
    mask = mask * 255
    mask = Image.fromarray(mask.astype(np.uint8))
    mask.save("mask.png")
