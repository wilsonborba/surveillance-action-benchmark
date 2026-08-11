from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class TrackState:
    center: np.ndarray
    age: int = 0


class CentroidTracker:
    def __init__(self, max_distance: float = 120.0, max_age: int = 20) -> None:
        self.max_distance = max_distance
        self.max_age = max_age
        self.next_track_id = 1
        self.tracks: Dict[int, TrackState] = {}

    def update(self, detections: List[dict]) -> List[dict]:
        centers = [np.array([(det["bbox"][0] + det["bbox"][2]) / 2.0, (det["bbox"][1] + det["bbox"][3]) / 2.0]) for det in detections]
        assigned_tracks = set()
        assigned_detections = set()

        for track_id, state in list(self.tracks.items()):
            best_index = None
            best_distance = self.max_distance
            for index, center in enumerate(centers):
                if index in assigned_detections:
                    continue
                distance = float(np.linalg.norm(center - state.center))
                if distance < best_distance:
                    best_distance = distance
                    best_index = index
            if best_index is None:
                state.age += 1
                if state.age > self.max_age:
                    del self.tracks[track_id]
                continue
            state.center = centers[best_index]
            state.age = 0
            detections[best_index]["track_id"] = track_id
            assigned_tracks.add(track_id)
            assigned_detections.add(best_index)

        for index, detection in enumerate(detections):
            if index in assigned_detections:
                continue
            track_id = self.next_track_id
            self.next_track_id += 1
            self.tracks[track_id] = TrackState(center=centers[index], age=0)
            detection["track_id"] = track_id
        return detections
