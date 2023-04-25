"""
Script for preprocessing the shapenet meshes and saving them as watertight meshes.
The source meshes are ShapeNetSem raw models.

Run:
    python watertight.py [-h] --src SRC [--dst DST] [--grasps GRASPS] [--start START]

Expected directory structure:
    SRC/0123.obj
    GRASPS/Couch_0123_1.168.h5
Other parameters:
    DST: will be 'SRC/watertight' if not specified
    START=0123 (obj id string)

If provided, only ids from GRASPS will be processed.
The objects will be converted in order, based on their id.
START will be the first id to process.
"""

import os
from argparse import ArgumentParser


def setup():
    os.system('git clone --recursive -j8 https://github.com/hjwdzh/Manifold.git')
    os.chdir('Manifold')
    os.makedirs('build', exist_ok=True)
    os.chdir('build')
    os.system('cmake .. -DCMAKE_BUILD_TYPE=Release')
    os.system('make')
    os.chdir('../..')


def cleanup():
    os.system('rm temp.watertight.obj')
    os.system('rm -rf Manifold')


def mesh2manifold(src, dst):
    tmp = 'temp.watertight.obj'
    os.system(f'Manifold/build/manifold {src} {tmp}')
    os.system(f'Manifold/build/simplify -i {tmp} -o {dst} -m -r 0.02')


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--src', type=str, required=True, help='path/to/models')
    argparser.add_argument('--dst', type=str, help='path/to/watertight')
    argparser.add_argument('--grasps', type=str, help='path/to/acronym/grasps')
    argparser.add_argument('--start', type=str, help='starting object id')
    args = argparser.parse_args()

    setup()

    if args.grasps:
        obj_ids = [filename.split('_')[1] for filename in os.listdir(args.grasps)]
    else:
        obj_ids = [filename.split('.')[0] for filename in os.listdir(args.src)]
    obj_ids.sort()

    out_path = os.path.join(args.src, 'watertight') if not args.dst else args.dst
    os.makedirs(out_path, exist_ok=True)

    for obj_id in obj_ids:
        filename = f'{obj_id}.obj'

        src_obj = os.path.join(args.src, filename)

        if not os.path.exists(src_obj):
            print(obj_id, 'not found in src-models. skipping', obj_id)
            continue

        if args.start and obj_id < args.start:  # we are comparing strings
            print(obj_id, 'is previous to "start". skipping', obj_id)
            continue

        dst_obj = os.path.join(out_path, filename)
        mesh2manifold(src_obj, dst_obj)

    cleanup()
