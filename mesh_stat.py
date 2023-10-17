import datetime
import json
from pathlib import Path

import numpy
import vedo

import debug_settings
from operator import itemgetter
from Mesh import Mesh, count_face_types

from utils import draw_plot, collect_mesh_filepaths


def gen_mesh_info_json(datadir, output_path=debug_settings.output_path):
    tmp_file_path_list = collect_mesh_filepaths(datadir)
    tmp_mesh_info_list = []
    counter = 0

    for a_file_path in tmp_file_path_list:
        mesh_obj_info = generate_stat_of_single_mesh_ali(a_file_path)
        tmp_mesh_info_list.append(mesh_obj_info)
        counter += 1
        print("Dealing with:" + str(counter), end="\r", flush=True)
    print(json.dumps(tmp_mesh_info_list))
    with open(output_path + "\\" + str(datetime.datetime.now().strftime("(%d_%m_%Y_%H_%M_%S)")) + ".json", 'w+') as f:
        json.dump(tmp_mesh_info_list, f)
    return tmp_mesh_info_list


def gen_stat(info_list_json, extreme_example_number=5):
    statistics = {"avg_vertices": numpy.average([x["num_vertices"] for x in info_list_json]),
                  "avg_faces": numpy.average([x["num_faces"] for x in info_list_json]),
                  "extreme_shapes": {"shape_least_vertices": sorted(info_list_json, key=itemgetter('num_vertices'))[
                                                             :extreme_example_number],
                                     "shape_most_vertices": sorted(info_list_json, key=itemgetter('num_vertices'),
                                                                   reverse=True)[:extreme_example_number],
                                     "shape_least_faces": sorted(info_list_json, key=itemgetter('num_faces'))[
                                                          :extreme_example_number],
                                     "shape_most_faces": sorted(info_list_json, key=itemgetter('num_faces'),
                                                                reverse=True)[:extreme_example_number]}}
    return statistics


def statistic_from_data(datadir, show_plot=False):
    tmp_mesh_info_list = gen_mesh_info_json(datadir)
    statistics = gen_stat(tmp_mesh_info_list)

    if show_plot:
        draw_plot(tmp_mesh_info_list, datadir=datadir, sub_attribute="num_vertices")
        draw_plot(tmp_mesh_info_list, datadir=datadir, sub_attribute="num_faces")
    print(statistics)
    return statistics


# Assignment 2-2, but quicker due to it reads the basic info from files
def statistic_from_file(json_info_dir):
    file_contents = {}
    with open(json_info_dir) as user_file:
        file_contents = user_file.read()

    tmp_mesh_info_list = json.loads(file_contents)
    draw_plot(tmp_mesh_info_list, datadir="", sub_attribute="num_vertices")
    draw_plot(tmp_mesh_info_list, datadir="", sub_attribute="num_faces")
    statistics = gen_stat(tmp_mesh_info_list)
    print(statistics)
    return statistics


def generate_stat_of_single_mesh_vedo(file_path):
    mesh_obj = vedo.Mesh(file_path)
    mesh_obj.compute_cell_vertex_count()

    rst = {"file_path": file_path,
           "class_of_shape": file_path.replace("\\" + Path(file_path).name, "").split("\\")[-1],
           "num_faces": len(mesh_obj.faces()),
           "num_vertices": mesh_obj.npoints,
           "type_faces": count_face_types(mesh_obj),
           "box": {'x': str(round(mesh_obj.bounds()[1] - mesh_obj.bounds()[0], 5)),
                   "y": str(round(mesh_obj.bounds()[3] - mesh_obj.bounds()[2], 5)),
                   "z": str(round(mesh_obj.bounds()[5] - mesh_obj.bounds()[4], 5)),
                   }}
    print(rst)
    return rst


def generate_stat_of_single_mesh_ali(file_path):
    mesh_obj = Mesh(file_path)
    mesh_info = mesh_obj.serialize()
    rst = {"file_path": mesh_info["filepath"],
           "class_of_shape": mesh_info["class"],
           "num_faces": mesh_info["faces"],
           "num_vertices": mesh_info["vertices"],
           "box": mesh_info["bounding_box"]}
    return rst
