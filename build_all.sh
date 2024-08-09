DIR=$(pwd)

cd $DIR/mycpp/ && mkdir -p build && cd build && cmake .. -DPYTHON_EXECUTABLE=$(which python) && make -j11
export MAX_JOBS=4
cd /kaolin && rm -rf build *egg* && pip install -e .
unset MAX_JOBS
cd $DIR/bundlesdf/mycuda && rm -rf build *egg* && pip install -e .

cd ${DIR}
