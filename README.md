# Face Detection and Alignment with OpenPose
The project modify from [OpenPose SDK](https://github.com/CMU-Perceptual-Computing-Lab/openpose.git). We use the sample of OpenPose and change cpu-render code which is in the ./src/openpose/utilities/keypoint.cpp file. 

# compile 
We perform face alignment with the pre-trained model trained by CMU perceptual computing lab. You can download the model with cmake command, shell script getModels.sh in models folder or download model from [this](https://pan.baidu.com/s/1gf6IRq7) and unzip model into model folder. 
[OpenCV](https://github.com/opencv/opencv.git) is a additional library. You must compile code with the library.
You can compile project following the steps:
1. cd build && cmake .. -DCMAKE_BUILD_LIBS=Release -DCMAKE_INSTALL_PREFIX=. -DOpenCV_DIR=/your/path/of/OpenCV
2. make -j 16 && make install
3. cd the root path of the project 
4. ./build/examples/openpose/openpose.bin --face --body_disable

# future work

