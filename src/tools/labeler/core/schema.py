"""
Schema definitions for pose estimation configurations.
Supports loading keypoint definitions from YAML files.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class KeypointDef:
    """Definition of a keypoint type."""
    name: str
    color: str = "#FF0000"
    index: int = 0
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "color": self.color,
            "index": self.index
        }


@dataclass
class Schema:
    """Schema defining keypoints, skeleton, and classes for a domain."""
    name: str
    version: str = "1.0"
    classes: dict[int, str] = field(default_factory=dict)
    keypoints: list[KeypointDef] = field(default_factory=list)
    skeleton: list[tuple[str, str]] = field(default_factory=list)
    flip_pairs: list[tuple[str, str]] = field(default_factory=list)
    
    def get_keypoint_names(self) -> list[str]:
        """Return ordered list of keypoint names."""
        return [kp.name for kp in self.keypoints]
    
    def get_keypoint_colors(self) -> dict[str, str]:
        """Return mapping of keypoint name to color."""
        return {kp.name: kp.color for kp in self.keypoints}
    
    def get_keypoint_index(self, name: str) -> int:
        """Get the index of a keypoint by name."""
        for i, kp in enumerate(self.keypoints):
            if kp.name == name:
                return i
        return -1
    
    def get_flip_idx(self) -> list[int]:
        """Generate flip_idx array for YOLO config."""
        names = self.get_keypoint_names()
        flip_idx = list(range(len(names)))
        
        for left, right in self.flip_pairs:
            left_idx = self.get_keypoint_index(left)
            right_idx = self.get_keypoint_index(right)
            if left_idx >= 0 and right_idx >= 0:
                flip_idx[left_idx] = right_idx
                flip_idx[right_idx] = left_idx
        
        return flip_idx
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "classes": self.classes,
            "keypoints": [kp.to_dict() for kp in self.keypoints],
            "skeleton": self.skeleton,
            "flip_pairs": self.flip_pairs
        }
    
    def to_yolo_config(self, dataset_path: str, train_path: str = "images/train", 
                       val_path: str = "images/val") -> dict:
        """Generate YOLO dataset config dictionary."""
        return {
            "path": dataset_path,
            "train": train_path,
            "val": val_path,
            "kpt_shape": [len(self.keypoints), 3],  # [num_keypoints, dims (x,y,visibility)]
            "flip_idx": self.get_flip_idx(),
            "names": self.classes,
            "kpt_names": {
                class_id: self.get_keypoint_names() 
                for class_id in self.classes.keys()
            }
        }
    
    @classmethod
    def from_yaml(cls, path: Path) -> "Schema":
        """Load schema from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: dict) -> "Schema":
        """Create schema from dictionary."""
        keypoints = []
        for i, kp_data in enumerate(data.get("keypoints", [])):
            if isinstance(kp_data, dict):
                keypoints.append(KeypointDef(
                    name=kp_data["name"],
                    color=kp_data.get("color", "#FF0000"),
                    index=i
                ))
            else:
                keypoints.append(KeypointDef(name=kp_data, index=i))
        
        skeleton = []
        for conn in data.get("skeleton", []):
            if isinstance(conn, (list, tuple)) and len(conn) == 2:
                skeleton.append((conn[0], conn[1]))
        
        flip_pairs = []
        for pair in data.get("flip_pairs", []):
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                flip_pairs.append((pair[0], pair[1]))
        
        classes = data.get("classes", {0: "object"})
        if isinstance(classes, dict):
            classes = {int(k): v for k, v in classes.items()}
        
        return cls(
            name=data.get("name", "Custom"),
            version=data.get("version", "1.0"),
            classes=classes,
            keypoints=keypoints,
            skeleton=skeleton,
            flip_pairs=flip_pairs
        )
    
    def save_yaml(self, path: Path) -> None:
        """Save schema to YAML file."""
        data = {
            "name": self.name,
            "version": self.version,
            "classes": self.classes,
            "keypoints": [
                {"name": kp.name, "color": kp.color}
                for kp in self.keypoints
            ],
            "skeleton": [list(conn) for conn in self.skeleton],
            "flip_pairs": [list(pair) for pair in self.flip_pairs]
        }
        
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def create_poultry_schema() -> Schema:
    """Hen pose schema — 10 keypoints covering head, spine, and legs.

    Index 2 (back_neck) is the painted identification spot used for ReID.
    """
    return Schema(
        name="Poultry Pose",
        version="2.0",
        classes={0: "hen"},
        keypoints=[
            # Index 0–1: head
            KeypointDef(name="beak",        color="#E63946", index=0),  # crimson red
            KeypointDef(name="crown",       color="#F4A261", index=1),  # warm orange
            # Index 2–3: neck / back
            KeypointDef(name="back_neck",   color="#E9C46A", index=2),  # golden ★ ReID anchor
            KeypointDef(name="middle_back", color="#52B788", index=3),  # sage green
            # Index 4–5: tail
            KeypointDef(name="tail_base",   color="#4895EF", index=4),  # blue
            KeypointDef(name="tail_tip",    color="#ADE8F4", index=5),  # light cyan
            # Index 6–7: left leg (warm tones)
            KeypointDef(name="left_hock",   color="#FF9F1C", index=6),  # amber
            KeypointDef(name="left_foot",   color="#FFBF69", index=7),  # light amber
            # Index 8–9: right leg (cool tones)
            KeypointDef(name="right_hock",  color="#9B5DE5", index=8),  # violet
            KeypointDef(name="right_foot",  color="#C77DFF", index=9),  # light violet
        ],
        skeleton=[
            ("beak",        "crown"),
            ("beak",        "back_neck"),
            ("back_neck",   "middle_back"),
            ("middle_back", "tail_base"),
            ("tail_base",   "tail_tip"),
            ("tail_base",   "left_hock"),
            ("tail_base",   "right_hock"),
            ("middle_back", "left_hock"),
            ("middle_back", "right_hock"),
            ("left_hock",   "left_foot"),
            ("right_hock",  "right_foot"),
        ],
        flip_pairs=[
            ("left_hock",  "right_hock"),
            ("left_foot",  "right_foot"),
        ]
    )


# Built-in schemas
BUILTIN_SCHEMAS = {
    "poultry": create_poultry_schema,
}


def get_builtin_schema(name: str) -> Optional[Schema]:
    """Get a built-in schema by name."""
    factory = BUILTIN_SCHEMAS.get(name.lower())
    if factory:
        return factory()
    return None


def list_builtin_schemas() -> list[str]:
    """List available built-in schema names."""
    return list(BUILTIN_SCHEMAS.keys())
