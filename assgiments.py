import vedo
from pathlib import Path
from Mesh import Mesh, count_face_types
from debug_settings import *
from mesh_resample import resampling_to_tenk, resample_core
from mesh_stat import gen_mesh_info_json, gen_stat, statistic_from_data, generate_stat_of_single_mesh_vedo, \
    generate_stat_of_single_mesh_ali
from utils import show_mesh, draw_plot


def assigment_1_1(mesh_obj_path, show_type="shaded"):
    show_mesh(mesh_obj_path, show_type)


def assignment_2_1(file_path):
    return generate_stat_of_single_mesh_ali(file_path)


def assignment_2_1_raw(file_path):
    return generate_stat_of_single_mesh_vedo(file_path)


# Assignment 2-2
def assignment_2_2(datadir, show_plot=False):
    return statistic_from_data(datadir, show_plot)


def assignment_2_3(file_path):
    #resampling_to_tenk(file_path)
    resample_core(file_path)
