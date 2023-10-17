import vedo
import debug_settings
from multiprocessing import Pool
from pathlib import Path

import mesh_stat
from mesh_stat import gen_mesh_info_json
from utils import collect_mesh_filepaths


def resampling_to_tenk(file_path, output_path=None):
    print("Re-sampling:"+file_path)
    print("Before re-sampling:")
    print(mesh_stat.generate_stat_of_single_mesh_vedo(file_path))
    mesh_obj = vedo.Mesh(file_path)
    file_name = Path(file_path).name
    counter = 0
    while mesh_obj.npoints < 10000:
        mesh_obj.subdivide(1, 4)
        counter += 1

    # After testing we find the decimation of 'city' and 'ClassicPiano' takes Enormous time to compute. Just skip them.
    if mesh_obj.npoints > 10000 and "City" not in file_path and "ClassicPiano" not in file_path:
        mesh_obj.decimate(n=10000)

    if output_path is not None:
        output_path = file_path + file_name
    else:
        output_path = debug_settings.output_path + file_name

    mesh_obj.write(output_path)
    print("After re-sampling:")
    print(mesh_stat.generate_stat_of_single_mesh_vedo(output_path))


def resample_core(file_path_dir, pool_size=20):
    mesh_paths = collect_mesh_filepaths(file_path_dir)
    with Pool(pool_size) as p:
        p.map(resampling_to_tenk, mesh_paths)


def show_outliers(mesh_dir):
    # status = statistic_from_file("E:\\MR\\src\\output\\(09_10_2023_23_47_19).json")
    status = mesh_stat.statistic_from_data(mesh_dir)
    for strange_meshes in status["extreme_shapes"]["shape_least_vertices"]:
        print(strange_meshes)
        tmp_mesh = vedo.Mesh(strange_meshes["file_path"])
        tmp_plt = vedo.Plotter()
        tmp_plt += tmp_mesh
        tmp_plt.show()
