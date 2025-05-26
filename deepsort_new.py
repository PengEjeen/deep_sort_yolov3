# Modified deepsort.py — Target ID recovery with IOU threshold and smooth tracking
import os
import cv2
import time
import argparse
import torch
import warnings
import json
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'thirdparty/fast-reid'))

from detector import build_detector
from deep_sort import build_tracker
from utils.draw import draw_boxes
from utils.parser import get_config
from utils.log import get_logger
from utils.io import write_results


def compute_iou(box1, box2):
    if box1 is None or box2 is None:
        return 0.0
    xi1, yi1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    xi2, yi2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - inter_area
    return inter_area / union if union > 0 else 0.0

def get_best_iou_track(outputs, target_bbox, return_iou=False):
    if target_bbox is None:
        return (None, 0.0) if return_iou else None
    best_iou = 0
    best_id = None
    for det in outputs:
        x1, y1, x2, y2 = det[:4]
        track_id = int(det[-1])
        iou = compute_iou([x1, y1, x2, y2], target_bbox)
        if iou > best_iou:
            best_iou = iou
            best_id = track_id
    if return_iou:
        return best_id, best_iou
    return best_id

class VideoTracker:
    def __init__(self, cfg, args, video_path):
        self.cfg = cfg
        self.args = args
        self.video_path = video_path
        self.logger = get_logger("root")
        self.first_frame_flag = True
        self.target_id = None
        self.last_known_bbox = None

        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        if args.cam != -1:
            self.vdo = cv2.VideoCapture(args.cam)
        else:
            self.vdo = cv2.VideoCapture(video_path)

        self.detector = build_detector(cfg, use_cuda=use_cuda, segment=args.segment)
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)

    def run(self):
        results = []
        idx_frame = 0
        with open('coco_classes.json', 'r') as f:
            idx_to_class = json.load(f)

        if not self.vdo.isOpened():
            raise IOError("Failed to open video")

        im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.args.save_path:
            os.makedirs(self.args.save_path, exist_ok=True)
            self.writer = cv2.VideoWriter(
                os.path.join(self.args.save_path, "results.avi"),
                cv2.VideoWriter_fourcc(*'MJPG'),
                20, (im_width, im_height))

        while self.vdo.grab():
            idx_frame += 1
            if idx_frame % self.args.frame_interval:
                continue

            _, ori_im = self.vdo.retrieve()
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)

            if self.args.segment:
                bbox_xywh, cls_conf, cls_ids, seg_masks = self.detector(im)
            else:
                bbox_xywh, cls_conf, cls_ids = self.detector(im)

            mask = cls_ids == 0  # person class
            bbox_xywh = bbox_xywh[mask]
            cls_conf = cls_conf[mask]
            cls_ids = cls_ids[mask]
            if bbox_xywh.shape[0] == 0:
                continue

            bbox_xywh[:, 2:] *= 1.2

            if self.args.segment:
                seg_masks = seg_masks[mask]
                outputs, mask_outputs = self.deepsort.update(bbox_xywh, cls_conf, cls_ids, im, seg_masks)
            else:
                outputs, _ = self.deepsort.update(bbox_xywh, cls_conf, cls_ids, im)

            if self.first_frame_flag and len(outputs) > 0:
                cv2.imshow("Select target", ori_im)
                cv2.waitKey(1)
                target_roi = cv2.selectROI("Select target", ori_im, False, False)
                cv2.destroyWindow("Select target")
                target_bbox = [target_roi[0], target_roi[1], target_roi[0] + target_roi[2], target_roi[1] + target_roi[3]]
                self.target_id = get_best_iou_track(outputs, target_bbox)
                self.last_known_bbox = target_bbox
                print(f"[INFO] Selected target ID: {self.target_id}")
                self.first_frame_flag = False
                continue

            bbox_tlwh = []
            filtered_outputs = []
            for det in outputs:
                if int(det[-1]) == self.target_id:
                    filtered_outputs.append(det)
                    self.last_known_bbox = det[:4]

            if len(filtered_outputs) == 0 and self.last_known_bbox is not None:
                new_id, best_iou = get_best_iou_track(outputs, self.last_known_bbox, return_iou=True)
                if best_iou > 0.4:
                    self.target_id = new_id
                    print(f"[INFO] Target temporarily lost. Reassigned to ID {self.target_id} (IOU={best_iou:.2f})")
                    for det in outputs:
                        if int(det[-1]) == self.target_id:
                            filtered_outputs.append(det)
                            self.last_known_bbox = det[:4]
                else:
                    print("[INFO] IOU too low to reassign. Skipping reassignment.")

            if len(filtered_outputs) > 0:
                def box_center(box):
                    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])

                smoothed_outputs = []
                for det in filtered_outputs:
                    if self.last_known_bbox is None:
                        smoothed_outputs.append(det)
                        continue
                    dist = np.linalg.norm(box_center(det[:4]) - box_center(self.last_known_bbox))
                    if dist < 300:
                        smoothed_outputs.append(det)
                    else:
                        print(f"[INFO] Skipped jumpy box with dist={dist:.2f}")

                if len(smoothed_outputs) > 0:
                    bbox_xyxy = np.array([det[:4] for det in smoothed_outputs])
                    identities = [int(det[-1]) for det in smoothed_outputs]
                    cls = [int(det[-2]) for det in smoothed_outputs]
                    names = [idx_to_class[str(label)] for label in cls]

                    ori_im = draw_boxes(ori_im, bbox_xyxy, names, identities)

                    for box in bbox_xyxy:
                        bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(box))

                    results.append((idx_frame - 1, bbox_tlwh, identities, cls))

            if self.args.display:
                cv2.imshow("test", ori_im)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if self.args.save_path:
                self.writer.write(ori_im)

        if self.args.save_path:
            write_results(os.path.join(self.args.save_path, "results.txt"), results, 'mot')

        self.vdo.release()
        if self.args.display:
            cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--VIDEO_PATH", type=str, default="demo.avi")
    parser.add_argument("--config_mmdetection", type=str, default="./configs/mmdet.yaml")
    parser.add_argument("--config_detection", type=str, default="./configs/mask_rcnn.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    parser.add_argument("--config_fastreid", type=str, default="./configs/fastreid.yaml")
    parser.add_argument("--fastreid", action="store_true")
    parser.add_argument("--mmdet", action="store_true")
    parser.add_argument("--segment", action="store_true")
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="./output/")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    cfg = get_config()

    cfg.USE_SEGMENT = args.segment
    cfg.USE_MMDET = args.mmdet
    cfg.USE_FASTREID = args.fastreid

    cfg.merge_from_file(args.config_mmdetection if args.mmdet else args.config_detection)
    cfg.merge_from_file(args.config_deepsort)
    if args.fastreid:
        cfg.merge_from_file(args.config_fastreid)

    tracker = VideoTracker(cfg, args, video_path=args.VIDEO_PATH)
    tracker.run()