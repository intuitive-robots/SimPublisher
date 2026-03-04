from __future__ import annotations

import json
from pathlib import Path
import yaml


def read_qr_alignment_data(filepath: str) -> dict:
    """Read QR alignment data from a YAML file."""
    
    with open(filepath, "r") as file:
        return yaml.safe_load(file)

def create_scene_config_file(scene_data: dict = None, output_filepath: Path = Path("Scene.json")) -> None:
    """Create a scene configuration JSON file.
    Args:
        scene_data (dict, optional): Scene configuration data. If None, a default scene is created.
        output_filepath (str): Path to the output JSON file. 
    """
    if scene_data is None:
        scene_data = {"x": 0, "y": 0, "z": 0, "rotX": 0, "rotY": 0, "rotZ": 0}
    output_filepath = Path(output_filepath)
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(output_filepath, "w") as file:
        json.dump(scene_data, file)
