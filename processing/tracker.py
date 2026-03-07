"""
Simple player tracker using IoU matching.

Maintains consistent player IDs across frames.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


class PlayerTracker:

    def __init__(self, iou_threshold=0.3, max_lost_frames=10):

        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames

        self.next_id = 0

        self.tracked_players = {}
        self.id_mapping = {}

    def compute_iou(self, box1, box2):

        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union = area1 + area2 - inter

        if union == 0:
            return 0

        return inter / union

    def update(self, detections):

        detections = list(detections)

        if len(detections) == 0:

            remove_ids = []

            for pid in list(self.tracked_players.keys()):

                self.tracked_players[pid]["lost"] += 1

                if self.tracked_players[pid]["lost"] > self.max_lost_frames:
                    remove_ids.append(pid)

            for pid in remove_ids:
                del self.tracked_players[pid]

            return {}, []

        if len(self.tracked_players) == 0:

            self.id_mapping = {}

            for i, box in enumerate(detections):

                self.tracked_players[self.next_id] = {
                    "bbox": box,
                    "lost": 0,
                }

                self.id_mapping[i] = self.next_id
                self.next_id += 1

            return self.id_mapping, detections

        tracked_ids = list(self.tracked_players.keys())
        tracked_boxes = [self.tracked_players[i]["bbox"] for i in tracked_ids]

        cost_matrix = np.zeros((len(tracked_boxes), len(detections)))

        for i, tbox in enumerate(tracked_boxes):
            for j, dbox in enumerate(detections):

                iou = self.compute_iou(tbox, dbox)
                cost_matrix[i, j] = 1 - iou

        t_idx, d_idx = linear_sum_assignment(cost_matrix)

        used_dets = set()
        matched_tracks = set()

        self.id_mapping = {}

        for ti, di in zip(t_idx, d_idx):

            iou = 1 - cost_matrix[ti, di]

            if iou >= self.iou_threshold:

                pid = tracked_ids[ti]

                self.tracked_players[pid]["bbox"] = detections[di]
                self.tracked_players[pid]["lost"] = 0

                self.id_mapping[di] = pid

                used_dets.add(di)
                matched_tracks.add(pid)

        for di, box in enumerate(detections):

            if di not in used_dets:

                self.tracked_players[self.next_id] = {
                    "bbox": box,
                    "lost": 0,
                }

                self.id_mapping[di] = self.next_id
                self.next_id += 1

        remove_ids = []

        for pid in list(self.tracked_players.keys()):

            if pid not in matched_tracks:

                self.tracked_players[pid]["lost"] += 1

                if self.tracked_players[pid]["lost"] > self.max_lost_frames:
                    remove_ids.append(pid)

        for pid in remove_ids:
            del self.tracked_players[pid]

        tracked_boxes = []

        for di in range(len(detections)):

            if di in self.id_mapping:

                pid = self.id_mapping[di]
                tracked_boxes.append(self.tracked_players[pid]["bbox"])

            else:
                tracked_boxes.append(detections[di])

        return self.id_mapping, tracked_boxes

    def get_detection_indices(self, selected_ids):

        reverse = {v: k for k, v in self.id_mapping.items()}

        indices = set()

        for pid in selected_ids:

            if pid in reverse:
                indices.add(reverse[pid])

        return indices