"""
Author: Wouter Van Gansbeke

Panoptic evaluation class for class agnostic models
Mostly copied from detectron2 with some modifications to make it compatible with our codebase
See https://github.com/facebookresearch/detectron2 for more details and license.
"""

import torch
import contextlib
import io
import itertools
import json
import logging
import os
import tempfile
from collections import OrderedDict
from PIL import Image
from tabulate import tabulate
import numpy as np

from detectron2.utils import comm
from detectron2.utils.file_io import PathManager
from detectron2.evaluation.evaluator import DatasetEvaluator
from termcolor import colored
from typing import Optional, List, Dict, Union

logger = logging.getLogger(__name__)


class PanopticEvaluatorAgnostic(DatasetEvaluator):
    """
    Evaluate Panoptic Quality metrics on COCO using PanopticAPI.
    It saves panoptic segmentation prediction in `output_dir`

    It contains a synchronize call and has to be called from all workers.
    """

    def __init__(self, output_dir: str = './output/predictions_panoptic/', meta: Optional[Dict] = None):
        """
        Args:
            dataset_name: name of the dataset
            output_dir: output directory to save results for evaluation.
        """
        self._metadata = meta
        self.thing_dataset_id_to_contiguous_id = self._metadata["thing_dataset_id_to_contiguous_id"]
        self.stuff_dataset_id_to_contiguous_id = self._metadata["stuff_dataset_id_to_contiguous_id"]
        self.panoptic_json = self._metadata['panoptic_json']
        self.panoptic_root = self._metadata['panoptic_root']
        self.label_divisor = 1
        self._thing_contiguous_id_to_dataset_id = {
            v: k for k, v in self._metadata["thing_dataset_id_to_contiguous_id"].items()
        }
        self._stuff_contiguous_id_to_dataset_id = {
            v: k for k, v in self._metadata["stuff_dataset_id_to_contiguous_id"].items()
        }

        self.class_agnostic = True
        if self.class_agnostic:
            # we need to modify the json of the ground truth to make it class agnostic and save it
            gt_json = PathManager.get_local_path(self.panoptic_json)
            gt_json_agnostic = gt_json.replace(".json", "_agnostic.json")
            self.panoptic_json = gt_json_agnostic
            if not os.path.exists(gt_json_agnostic):
                with open(gt_json, "r") as f:
                    json_data = json.load(f)
                for anno in json_data["annotations"]:
                    for seg in anno["segments_info"]:
                        seg["category_id"] = 1
                json_data['categories'] = [{'id': 1, 'name': 'object', 'supercategory': 'object', 'isthing': 1}]
                with PathManager.open(gt_json_agnostic, "w") as f:
                    f.write(json.dumps(json_data))

        self._output_dir = output_dir
        if self._output_dir is not None:
            PathManager.mkdirs(self._output_dir)

    def reset(self):
        self._predictions = []

    def _convert_category_id(self, segment_info: dict) -> dict:
        isthing = segment_info.pop("isthing", None)
        if isthing is None:
            # the model produces panoptic category id directly. No more conversion needed
            return segment_info
        if isthing is True:
            segment_info["category_id"] = self._thing_contiguous_id_to_dataset_id[
                segment_info["category_id"]
            ]
        else:
            segment_info["category_id"] = self._stuff_contiguous_id_to_dataset_id[
                segment_info["category_id"]
            ]
        return segment_info

    def process(
        self,
        file_names: List[str],
        image_ids: List[int],
        outputs: Dict[str, Union[torch.Tensor, np.ndarray, dict]],
    ):
        from panopticapi.utils import id2rgb

        for file_name, image_id, output in zip(file_names, image_ids, outputs):
            panoptic_img, segments_info = output["panoptic_seg"]
            if isinstance(panoptic_img, torch.Tensor):
                panoptic_img = panoptic_img.cpu().numpy()

            for seg_id in segments_info:
                seg_id['category_id'] = 1
                seg_id['isthing'] = True

            file_name = os.path.basename(file_name)
            file_name_png = os.path.splitext(file_name)[0] + ".png"
            with io.BytesIO() as out:
                Image.fromarray(id2rgb(panoptic_img)).save(out, format="PNG")
                if not self.class_agnostic:
                    segments_info = [self._convert_category_id(x) for x in segments_info]
                self._predictions.append(
                    {
                        "image_id": image_id,
                        "file_name": file_name_png,
                        "png_string": out.getvalue(),
                        "segments_info": segments_info,
                    }
                )

    def evaluate(self):
        comm.synchronize()

        self._predictions = comm.gather(self._predictions)
        self._predictions = list(itertools.chain(*self._predictions))
        if not comm.is_main_process():
            return

        # PanopticApi requires local files
        gt_json = PathManager.get_local_path(self.panoptic_json)
        gt_folder = PathManager.get_local_path(self.panoptic_root)

        with tempfile.TemporaryDirectory(prefix="panoptic_eval") as pred_dir:
            logger.info("Writing all panoptic predictions to {} ...".format(pred_dir))
            print(colored('Writing all panoptic predictions to {}...'.format(pred_dir), 'blue'))
            for p in self._predictions:
                with open(os.path.join(pred_dir, p["file_name"]), "wb") as f:
                    f.write(p.pop("png_string"))

            with open(gt_json, "r") as f:
                json_data = json.load(f)
            json_data["annotations"] = self._predictions

            output_dir = self._output_dir or pred_dir
            predictions_json = os.path.join(output_dir, "predictions.json")
            with PathManager.open(predictions_json, "w") as f:
                f.write(json.dumps(json_data))

            with contextlib.redirect_stdout(io.StringIO()):
                pq_res, pq_stat_per_cat, num_preds = pq_compute(
                    gt_json,
                    PathManager.get_local_path(predictions_json),
                    gt_folder=gt_folder,
                    pred_folder=pred_dir,
                )

        res = {}
        res["PQ"] = 100 * pq_res["All"]["pq"]
        res["SQ"] = 100 * pq_res["All"]["sq"]
        res["RQ"] = 100 * pq_res["All"]["rq"]
        res["PQ_th"] = 100 * pq_res["Things"]["pq"]
        res["SQ_th"] = 100 * pq_res["Things"]["sq"]
        res["RQ_th"] = 100 * pq_res["Things"]["rq"]
        if "Stuff" in pq_res:
            res["PQ_st"] = 100 * pq_res["Stuff"]["pq"]
            res["SQ_st"] = 100 * pq_res["Stuff"]["sq"]
            res["RQ_st"] = 100 * pq_res["Stuff"]["rq"]

        results = OrderedDict({"panoptic_seg": res})

        precision = pq_stat_per_cat[1].tp / (pq_stat_per_cat[1].tp + pq_stat_per_cat[1].fp + 1e-8)
        recall = pq_stat_per_cat[1].tp / (pq_stat_per_cat[1].tp + pq_stat_per_cat[1].fn + 1e-8)
        print('')
        print('precision: ', precision*100)
        print('recall: ', recall*100)
        print('found {} predictions'.format(num_preds))
        print(colored(get_table(pq_res), 'yellow'))
        return results


def pq_compute(
    gt_json_file: str,
    pred_json_file: str,
    gt_folder=None,
    pred_folder=None,
):
    from panopticapi.evaluation import pq_compute_multi_core

    with open(gt_json_file, 'r') as f:
        gt_json = json.load(f)
    with open(pred_json_file, 'r') as f:
        pred_json = json.load(f)

    if gt_folder is None:
        gt_folder = gt_json_file.replace('.json', '')
    if pred_folder is None:
        pred_folder = pred_json_file.replace('.json', '')
    categories = {el['id']: el for el in gt_json['categories']}

    if not os.path.isdir(gt_folder):
        raise Exception("Folder {} with ground truth segmentations doesn't exist".format(gt_folder))
    if not os.path.isdir(pred_folder):
        raise Exception("Folder {} with predicted segmentations doesn't exist".format(pred_folder))

    pred_annotations = {el['image_id']: el for el in pred_json['annotations']}
    matched_annotations_list = []
    for gt_ann in gt_json['annotations']:
        image_id = gt_ann['image_id']
        if image_id not in pred_annotations:
            # raise Exception('no prediction for the image with id: {}'.format(image_id))
            continue
        matched_annotations_list.append((gt_ann, pred_annotations[image_id]))

    pq_stat = pq_compute_multi_core(matched_annotations_list, gt_folder, pred_folder, categories)

    metrics = [("All", None), ("Things", True)]
    results = {}
    for name, isthing in metrics:
        results[name], per_class_results = pq_stat.pq_average(categories, isthing=isthing)
        if name == 'All':
            results['per_class'] = per_class_results

    return results, pq_stat.pq_per_cat, len(pred_annotations)


def get_table(pq_res: dict):
    headers = ["", "PQ", "SQ", "RQ", "#categories"]
    data = []
    for name in ["All", "Things", "Stuff"]:
        if name not in pq_res:
            continue
        row = [name] + [pq_res[name][k] * 100 for k in ["pq", "sq", "rq"]] + [pq_res[name]["n"]]
        data.append(row)
    table = tabulate(
        data, headers=headers, tablefmt="pipe", floatfmt=".3f", stralign="center", numalign="center"
    )
    return table
