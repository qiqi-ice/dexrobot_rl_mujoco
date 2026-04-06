import os
from dataclasses import dataclass
import numpy as np


@dataclass
class ObjectSpec:
    geom_type: str
    size: tuple
    mass: float
    rgba: tuple
    pos: tuple
    euler: tuple


OBJECT_LIBRARY = {
    "box": {
        "geom_type": "box",
        "size_low": np.array([0.02, 0.02, 0.02]),
        "size_high": np.array([0.04, 0.04, 0.05]),
        "mass": (0.02, 0.08),
    },
    "thin_box": {
        "geom_type": "box",
        "size_low": np.array([0.04, 0.015, 0.008]),
        "size_high": np.array([0.06, 0.025, 0.015]),
        "mass": (0.02, 0.05),
    },
    "cylinder": {
        "geom_type": "cylinder",
        "size_low": np.array([0.018, 0.03]),
        "size_high": np.array([0.03, 0.055]),
        "mass": (0.03, 0.09),
    },
    "capsule": {
        "geom_type": "capsule",
        "size_low": np.array([0.015, 0.03]),
        "size_high": np.array([0.025, 0.06]),
        "mass": (0.02, 0.08),
    },
    "ellipsoid": {
        "geom_type": "ellipsoid",
        "size_low": np.array([0.02, 0.02, 0.025]),
        "size_high": np.array([0.04, 0.035, 0.05]),
        "mass": (0.02, 0.07),
    },
}


def sample_object_spec(rng, object_types=None):
    candidates = object_types or list(OBJECT_LIBRARY.keys())
    key = candidates[int(rng.integers(0, len(candidates)))]
    cfg = OBJECT_LIBRARY[key]
    size = rng.uniform(cfg["size_low"], cfg["size_high"])
    mass = float(rng.uniform(*cfg["mass"]))
    color = tuple(rng.uniform([0.2, 0.4, 0.4, 1.0], [0.9, 0.9, 0.95, 1.0]))
    pos = (
        float(rng.uniform(0.20, 0.32)),
        float(rng.uniform(-0.10, 0.10)),
        float(0.84 + rng.uniform(-0.01, 0.01)),
    )
    euler = (
        float(rng.uniform(-0.25, 0.25)),
        float(rng.uniform(-0.25, 0.25)),
        float(rng.uniform(-np.pi, np.pi)),
    )
    return ObjectSpec(
        geom_type=cfg["geom_type"],
        size=tuple(float(v) for v in size),
        mass=mass,
        rgba=color,
        pos=pos,
        euler=euler,
    )


def _fmt(values):
    return " ".join(f"{v:.6f}" for v in values)


def build_grasp_scene_xml(output_path, object_spec, repo_root=None):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    xml = f'''<?xml version="1.0" encoding="utf-8"?>
<mujoco model="dexrobot_rl_grasp_scene">
    <include file="../dexrobot_mujoco/parts/defaults.xml" />
    <include file="../dexrobot_mujoco/scenes/scene_sim/topfloor_scene.xml" />
    <include file="../dexrobot_mujoco/scenes/furniture_sim/simpleTable/simpleTable_asset.xml" />
    <include file="../dexrobot_mujoco/parts/dexhand021_right_realmanRobot_asset.xml" />

    <worldbody>
        <camera name="rgb_camera" pos="0 1.3 1" euler="-1.7 0 3.1415926" fovy="60"/>
        <body name="dexhand021_with_realmanRobot" pos="-0.35 0 0.76" euler="0 0 3.14159265">
            <include file="../dexrobot_mujoco/parts/dexhand021_right_realmanRobot_body.xml" />
        </body>
        <body name="scenetable" pos="0 0 0" euler="0 0 1.57">
            <include file="../dexrobot_mujoco/scenes/furniture_sim/simpleTable/simpleMarbleTable_body.xml" />
        </body>
        <body name="grasp_object" pos="{_fmt(object_spec.pos)}" euler="{_fmt(object_spec.euler)}">
            <inertial pos="0 0 0" mass="{object_spec.mass:.6f}" diaginertia="0.00002 0.00002 0.00002" />
            <geom name="grasp_object_geom" type="{object_spec.geom_type}" size="{_fmt(object_spec.size)}" rgba="{_fmt(object_spec.rgba)}" />
            <freejoint name="grasp_object_joint" />
        </body>
    </worldbody>

    <include file="../dexrobot_mujoco/parts/dexhand021_right_realmanRobot_actuator.xml" />
    <include file="../dexrobot_mujoco/parts/dexhand021_right_realmanRobot_sensor.xml" />
    <include file="../dexrobot_mujoco/parts/dexhand021_right_realmanRobot_contact.xml" />
</mujoco>
'''
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(xml)
    return output_path
