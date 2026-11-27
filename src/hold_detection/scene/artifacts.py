"""Shared paths and readers for a completed wall analysis."""

import json


SCENE_FILENAME = "scene.json"
SCENE_SCHEMA_VERSION = "5.0"


def frame_record_paths(run):
    return sorted((run / "inference").glob("*/holds.json"))


def read_scene(run):
    return json.loads((run / SCENE_FILENAME).read_text())


def read_single_image_size(run):
    frames = frame_record_paths(run)
    if len(frames) != 1:
        raise ValueError("Expected exactly one processed image in 2D mode")
    return json.loads(frames[0].read_text())["processed_size_wh"]
