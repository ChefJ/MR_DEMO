import datetime
import json

import argparse
import os

import numpy
import vedo
import matplotlib.pyplot as plt

from debug_settings import DATASET_PATH, MESH_APPENDIX
from pathlib import Path


def collect_mesh_filepaths(file_dir):
    print("Collecting files with extension " + ",".join(MESH_APPENDIX) + " under " + file_dir)
    rst = []
    for subdir, dirs, files in os.walk(file_dir):
        for file in files:
            if file.endswith(MESH_APPENDIX):
                if os.path.isfile(os.path.join(subdir, file)):
                    rst.append(os.path.join(subdir, file))
    print("Done. " + str(len(rst)) + " files found.")
    return rst


def show_mesh(mesh_obj_path, show_type="wireframe"):
    mesh_obj = vedo.Mesh(mesh_obj_path).c("blue")
    if show_type == "wireframe":
        mesh_obj.wireframe(True)
        vedo.show(mesh_obj)
        return
    elif show_type == "shaded":
        vedo.show(mesh_obj)
        return

    mesh_obj_wireframe = mesh_obj.clone()
    mesh_obj_wireframe.wireframe(True).c("yellow")
    tmp_plt = vedo.Plotter()
    tmp_plt += mesh_obj
    tmp_plt += mesh_obj_wireframe
    tmp_plt.show()


def draw_plot(info_list, datadir, sub_attribute="num_vertices", bin_size=2000, tick_size=10000,
                x_axis_lable="Number of ", y_axis_lable="Count"):
    x_axis_lable += sub_attribute
    data_list = [x[sub_attribute] for x in info_list]
    plt.hist(data_list, bins=numpy.arange(0, max(data_list), bin_size))
    plt.xticks(numpy.arange(0, max(data_list), tick_size))
    plt.title(
        "Average:" + str(numpy.average(data_list)) + " Min:" + str(min(data_list)) + " Max:" + str(max(data_list)))
    fig = plt.gcf()
    fig.canvas.manager.set_window_title('Original Data Set' + str(datadir))

    fig.set_size_inches(18.5, 10.5)
    plt.xlabel(x_axis_lable)
    plt.ylabel(y_axis_lable)
    plt.autoscale()
    plt.show()


def find_missing_files():
    r_files = collect_mesh_filepaths("E:\\MR\\src\\output")
    r_name_list = [Path(ap).name for ap in r_files]
    original_files = collect_mesh_filepaths(DATASET_PATH)
    for aof in original_files:
        if Path(aof).name not in r_name_list:
            print(aof)
