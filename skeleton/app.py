from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from copy import deepcopy

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from point_cloud import PointCloud, load_obj_file
from smoothing import smooth_n_times, smooth_n_times_cotan



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
    original_point_cloud: PointCloud | None = None
    point_cloud: PointCloud | None = None
    show_points: bool = False
    show_mesh: bool = True
    smoothing_iter: int = 1
    lmbda: float = 0.1
    laplace_operator_idx: int = 0
    laplace_operator_items: list[str] = None

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
        self.state.laplace_operator_items = ["uniform", "cotangent"]

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
        _, self.state.laplace_operator_idx = psim.Combo("Laplace operator", self.state.laplace_operator_idx, self.state.laplace_operator_items)
        _, self.state.smoothing_iter = psim.SliderInt("Iterations", self.state.smoothing_iter, 1, 20)
        _, self.state.lmbda = psim.SliderFloat("Lambda", self.state.lmbda, 0, 1)
        if psim.Button("Smooth"):
            if self.state.laplace_operator_items[self.state.laplace_operator_idx] == "uniform":
                self.state.point_cloud.vertices = smooth_n_times(
                    self.state.smoothing_iter,
                    self.state.point_cloud.vertices,
                    self.state.point_cloud.faces,
                    h=self.state.lmbda
                )
            elif self.state.laplace_operator_items[self.state.laplace_operator_idx] == "cotangent":
                self.state.point_cloud.vertices = smooth_n_times_cotan(
                    self.state.smoothing_iter,
                    self.state.point_cloud.vertices,
                    self.state.point_cloud.faces,
                    h=self.state.lmbda
                )
            else:
                raise ValueError(f"Unknown laplace operator: {self.state.laplace_operator_items[self.state.laplace_operator_idx]}")
            ps.remove_all_structures()
            self._register_point_cloud(self.state.original_point_cloud, "original_point_cloud", see_through=True)
            self._register_point_cloud(self.state.point_cloud, "smoothed_point_cloud")

    def load_selected_file(self):
        path = self.state.files[self.state.selected_idx]
        point_cloud = load_obj_file(path)

        self.state.point_cloud = point_cloud
        self.state.original_point_cloud = deepcopy(point_cloud)
        self.state.constraints = None
        self.state.grid = None
        self.state.normals_flipped = False

        ps.remove_all_structures()
        self._register_point_cloud(point_cloud, "original_point_cloud")

        ps.reset_camera_to_home_view()

    def _register_point_cloud(self, point_cloud: PointCloud, point_cloud_name: str = "point_cloud", see_through: bool = False):
        mesh = ps.register_surface_mesh(
            point_cloud_name,
            point_cloud.vertices,
            point_cloud.faces,
            enabled=self.state.show_mesh,
        )
        
        if see_through:
            mesh.set_transparency(0.3)
            mesh.set_color([0.5, 0.5, 0.5])
        else:
            mesh.set_transparency(1)
            mesh.set_color([0.3, 0.8, 0.5])



