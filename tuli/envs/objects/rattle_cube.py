import numpy as np

from robosuite.models.objects import MujocoGeneratedObject
from robosuite.utils.mjcf_utils import (
    array_to_string,
    new_body,
    new_geom,
    new_joint,
    new_site,
)


# Material presets calibrated from physical properties.
# Table surface: Oak hardwood (E=12 GPa, ν=0.40).
# See docs/material_calibration.md for derivation and references.
CUBE_MATERIALS = {
    "soft_cube": {
        "density": 50.0,
        "solref": "0.04 0.52",
        "friction": "0.57 0.005 0.0001",
        "rgba": (0.9, 0.85, 0.2, 1.0),       # yellow-ish sponge
    },
    "plastic_cube": {
        "density": 1050.0,
        "solref": "0.006 0.28",
        "friction": "0.40 0.002 0.0001",
        "rgba": (0.2, 0.6, 0.9, 1.0),         # blue plastic
    },
    "metal_cube": {
        "density": 7850.0,
        "solref": "0.0024 0.16",
        "friction": "0.30 0.001 0.00005",
        "rgba": (0.7, 0.7, 0.75, 1.0),        # metallic grey
    },
}


class CubeObject(MujocoGeneratedObject):

    def __init__(
        self,
        name,
        material="plastic_cube",
        size=0.04,
        joints=None,
    ):
        assert material in CUBE_MATERIALS, \
            f"material must be one of {list(CUBE_MATERIALS.keys())}, got {material!r}"

        self.size = size
        mat = CUBE_MATERIALS[material]
        self.cube_density = mat["density"]
        self.cube_solref = mat["solref"]
        self.cube_friction = mat["friction"]
        self.cube_rgba = mat["rgba"]

        if joints == "default":
            self.joint_specs = [self.get_joint_attrib_template()]
        elif joints is None:
            self.joint_specs = []
        else:
            self.joint_specs = joints

        for i, joint_spec in enumerate(self.joint_specs):
            if "name" not in joint_spec:
                joint_spec["name"] = f"joint{i}"

        super().__init__(obj_type="all", duplicate_collision_geoms=False)

        self._name = name
        self._obj = self._get_object_subtree()
        self._get_object_properties()

    def _get_object_subtree(self):
        obj = new_body(name="root")

        for joint_spec in self.joint_specs:
            obj.append(new_joint(**joint_spec))

        site_attr = self.get_site_attrib_template()
        site_attr["name"] = "default_site"
        site_attr["rgba"] = "1 0 0 0"
        obj.append(new_site(**site_attr))

        col_attr = self.get_collision_attrib_template()
        col_attr.update({
            "name": "cube",
            "type": "box",
            "size": array_to_string([self.size, self.size, self.size]),
            "pos": "0 0 0",
            "density": str(self.cube_density),
            "solref": self.cube_solref,
            "friction": self.cube_friction,
        })
        obj.append(new_geom(**col_attr))

        vis_attr = self.get_visual_attrib_template()
        vis_attr.update({
            "name": "cube_vis",
            "type": "box",
            "size": array_to_string([self.size, self.size, self.size]),
            "pos": "0 0 0",
            "rgba": array_to_string(self.cube_rgba),
        })
        obj.append(new_geom(**vis_attr))

        return obj

    @property
    def bottom_offset(self):
        return np.array([0.0, 0.0, -self.size])

    @property
    def top_offset(self):
        return np.array([0.0, 0.0, self.size])

    @property
    def horizontal_radius(self):
        return np.sqrt(2) * self.size

    def get_bounding_box_half_size(self):
        return np.array([self.size, self.size, self.size])
