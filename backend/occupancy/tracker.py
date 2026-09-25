"""
HVEAC Control Center - Person Tracker
Maintains identities across frames, prevents exploding tracks, and purges stale IDs.
"""

import time
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger("hveac.tracker")


class PersonTracker:
    def __init__(self, max_stale_seconds: float = 2.0):
        self.max_stale_seconds = max_stale_seconds
        # track_id -> {"bbox": (x1, y1, x2, y2), "confidence": conf, "last_seen": float, "hits": int}
        self.active_tracks: Dict[int, Dict[str, Any]] = {}
        self._fallback_id_counter = 1000

    def update(self, detections: List[Dict[str, Any]]) -> Tuple[List[int], int]:
        """
        Updates track state from current frame detections.
        Returns: (active_track_ids, active_track_count)
        """
        now = time.time()
        current_frame_ids = set()

        for det in detections:
            track_id = det.get("track_id")

            # If tracker hasn't assigned an ID yet, allocate fallback
            if track_id is None:
                track_id = self._match_or_allocate_fallback(det, now)
                det["track_id"] = track_id

            current_frame_ids.add(track_id)

            if track_id in self.active_tracks:
                self.active_tracks[track_id]["bbox"] = (det["x1"], det["y1"], det["x2"], det["y2"])
                self.active_tracks[track_id]["confidence"] = det["confidence"]
                self.active_tracks[track_id]["last_seen"] = now
                self.active_tracks[track_id]["hits"] += 1
            else:
                self.active_tracks[track_id] = {
                    "bbox": (det["x1"], det["y1"], det["x2"], det["y2"]),
                    "confidence": det["confidence"],
                    "first_seen": now,
                    "last_seen": now,
                    "hits": 1
                }

        # Purge stale tracks that haven't been seen within max_stale_seconds
        stale_ids = [
            tid for tid, data in self.active_tracks.items()
            if (now - data["last_seen"]) > self.max_stale_seconds
        ]
        for tid in stale_ids:
            del self.active_tracks[tid]

        # The active count in the current frame or recently confirmed tracks
        active_ids = sorted(list(current_frame_ids))
        return active_ids, len(active_ids)

    def _match_or_allocate_fallback(self, det: Dict[str, Any], now: float) -> int:
        """
        Simple spatial IoU matching for detections without YOLO tracker ID.
        """
        bbox = (det["x1"], det["y1"], det["x2"], det["y2"])
        best_id = None
        best_iou = 0.35  # Minimum threshold

        for tid, data in self.active_tracks.items():
            if now - data["last_seen"] < 0.5:
                iou = self._compute_iou(bbox, data["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_id = tid

        if best_id is not None:
            return best_id

        self._fallback_id_counter += 1
        return self._fallback_id_counter

    @staticmethod
    def _compute_iou(boxA, boxB) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        denominator = float(boxAArea + boxBArea - interArea)
        return interArea / denominator if denominator > 0 else 0.0

    def reset(self):
        """
        Clears all active track records.
        """
        self.active_tracks.clear()
        self._fallback_id_counter = 1000
        logger.info("[TRACKER RESET] Cleared all active identities and tracking history.")
