from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from point_cloud import PointCloud, load_obj_file
from smoothing import smooth_n_times



MESHES = Path(__file__).resolve().parent.parent / "meshes/obj_data"


def timed(func):
    def wrapper(*args, **kwargs):
        start = perf_counter()
        res = func(*args, **kwargs)
        end = perf_counter()
        print(f'Execution time: {end - start}')
        return res
    return wrapper


@dataclass
class PSState:
    files: list[Path]
    selected_idx: int = 0
    point_cloud: PointCloud | None = None
    show_points: bool = False
    show_mesh: bool = True
    smoothing_iter: int = 1

class PSApp:
    state: PSState

    def __init__(self):
        self.state = PSState(files=sorted(MESHES.glob("*.obj")))
        self.pointcloud_handle = None
        self.normal_handle = None
        self.bbox_handle = None
        self.constraint_handle = None
        self.grid_handle = None
        self.mesh_handle = None

    def run(self):
        ps.init()
        ps.set_user_callback(self.callback)
        ps.show()

    def callback(self):
        self._draw_file_loader()
        self._draw_display_controls()
        self.draw_laplacian_smoothing()

    def _draw_file_loader(self):
        psim.TextUnformatted("Mesh file loader")

        if len(self.state.files) == 0:
            psim.TextUnformatted(f"No .off files found in {MESHES}")
            return

        file_names = [path.name for path in self.state.files]
        changed, new_idx = psim.Combo("File", self.state.selected_idx, file_names)

        if changed:
            self.state.selected_idx = new_idx

        if psim.Button("Load selected Mesh"):
            self.load_selected_file()

        if self.state.point_cloud is not None:
            psim.TextUnformatted(f"Loaded: {self.state.point_cloud.name}")

    def _draw_display_controls(self):
        changed_points, self.state.show_points = psim.Checkbox(
            "Show vertices",
            self.state.show_points,
        )

        if changed_points and self.pointcloud_handle is not None:
            self.pointcloud_handle.set_enabled(self.state.show_points)

        changed_mesh, self.state.show_mesh = psim.Checkbox(
            "Show mesh",
            self.state.show_mesh,
        )

        if changed_mesh and self.mesh_handle is not None:
            self.mesh_handle.set_enabled(self.state.show_mesh)

    def draw_laplacian_smoothing(self):
        if self.state.point_cloud is None:
            psim.TextUnformatted("No point cloud loaded")
            return

        psim.TextUnformatted("Laplacian smoothing")
        psim.Separator()
        n_smooth = 1
        changed, self.state.smoothing_iter = psim.SliderInt("Iterations", self.state.smoothing_iter, 1, 100)
        if psim.Button("Smooth"):
            self.state.point_cloud.vertices = smooth_n_times(
                self.state.smoothing_iter,
                self.state.point_cloud.vertices,
                self.state.point_cloud.faces,
                h=1.0,
            )
            ps.remove_all_structures()
            self._register_point_cloud(self.state.point_cloud)

    def load_selected_file(self):
        path = self.state.files[self.state.selected_idx]
        point_cloud = load_obj_file(path)

        self.state.point_cloud = point_cloud
        self.state.constraints = None
        self.state.grid = None
        self.state.normals_flipped = False

        ps.remove_all_structures()
        self._register_point_cloud(point_cloud)

        ps.reset_camera_to_home_view()

    def _register_point_cloud(self, point_cloud: PointCloud):
        point_radius = 0.003 * point_cloud.bbox_diagonal

        if point_cloud.name == "cat.off":
            point_radius = 0.01

        self.pointcloud_handle = ps.register_point_cloud(
            "vertices",
            point_cloud.vertices,
            radius=point_radius,
            enabled=self.state.show_points,
        )

        ps.register_surface_mesh(
            "point_cloud_mesh",
            point_cloud.vertices,
            point_cloud.faces,
            enabled=self.state.show_mesh,
        )



