
 

#!/bin/bash

HERE=$(pwd)
src_path=$1  # shapenet/models/  (input path)
dst_path=$2  # shapenet/watertight/  (output path)

# strip last slash
src_path=${src_path%/}
dst_path=${dst_path%/}

# setup
git clone --recursive -j8 https://github.com/hjwdzh/Manifold.git
cd Manifold
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make

# convert files
cd $HERE
mkdir -p $dst_path
for src_file in $src_path/*.obj; do
  filename=$(basename $src_file)
  dst_file=$dst_path/$filename

  Manifold/build/manifold $src_file temp.watertight.obj
  Manifold/build/simplify -i temp.watertight.obj -o $dst_file -m -r 0.02
done

# cleanup
rm temp.watertight.obj
rm -rf Manifold