import numpy as np

from robosuite.models.objects import MujocoGeneratedObject
from robosuite.utils.mjcf_utils import (
    array_to_string,
    new_body,
    new_geom,
    new_joint,
    new_site,
)


MATERIALS = {
    "hard": {
        "cube_density": 500.0,
        "cube_size": 0.04,
        "cube_solref": (0.006, 0.28),
        "cube_friction": (0.40, 0.002, 0.0001),
        "cube_rgba": (0.2, 0.6, 0.9, 1.0),
        "table_solref": (0.006, 0.28),
        "table_friction": (1.0, 0.005, 0.0001),
    },
    "soft": {
        "cube_density": 150.0,
        "cube_size": 0.04,
        "cube_solref": (0.02, 1.0),
        "cube_friction": (0.40, 0.002, 0.0001),
        "cube_rgba": (0.9, 0.7, 0.3, 1.0),
        "table_solref": (0.02, 1.0),
        "table_friction": (1.0, 0.005, 0.0001),
    },
}


class CubeObject(MujocoGeneratedObject):

    def __init__(self, name, material="hard", joints=None):
        mat = MATERIALS[material]
        self.material = material
        self.size = mat["cube_size"]
        self.cube_density = mat["cube_density"]
        self.cube_solref = array_to_string(mat["cube_solref"])
        self.cube_friction = array_to_string(mat["cube_friction"])
        self.cube_rgba = mat["cube_rgba"]

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
