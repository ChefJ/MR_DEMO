import argparse

import utils
from debug_settings import *
from utils import *
from mesh_stat import *
from mesh_resample import *
from assgiments import *



if __name__ == '__main__':
    args = None
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--datadir", help="The root directory of mesh files.Example: MR_3Guys.py /root/datadir/",
                            default=DATASET_PATH, type=str)
        args = parser.parse_args()

    except Exception as e:
        print("ERR PARSING THE ARGS")
        print(str(e))

    mesh_paths = utils.collect_mesh_filepaths(args.datadir)
    assigment_1_1(r"E:\MR\DATASET\ShapeDatabase_INFOMR-master\ShapeDatabase_INFOMR-master\Door\D01121.obj", "wireframe")
    # assignment_2_1_raw(mesh_paths[0])
    # print(assignment_2_1(mesh_paths[0]))
    # assignment_2_2(args.datadir, True)
    # assignment_2_3(mesh_paths)
