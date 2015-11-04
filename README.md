# dense-flow
Dense Optical Flow extraction rewritten from https://github.com/wanglimin/dense_flow

## OpenCV 3.0
The tool has been rewritten to be compatible with OpenCV 3.0. Make sure you compile OpenCV with CUDA capabilities.
If you installed OpenCV to a non-default path, please edit Makefile.config and edit the paths correspondingly.

## Serialization
In case of long video files, generating thousands of jpegs may be inconvenient for copying. We added the possibility to serialize them into one file. If you want to use this feature, please use 'make serialize'.

Serialization requires boost library!

### File format
All jpegs are encoded as a string (using OpenCV's imencode). The resulting vector of string is serialized using boost. unpacker.cc provides an example of how to extract the jpegs.

### Chunks
After 10000 files, the serialized files are written to disk, and a new chunk is started.

## Compilation
To compile, create 'build' directory, and run 'make'.

## Usage
Example:
./denseFlow_gpu --vidFile="video.mp4" --xFlowFile="flow_x" --yFlowFile="flow_y" --imgFile="im" --bound=16 --type=2 --device_id=0 --step=10

Reads from file video.mp4, saving the Dense Optical flow with corresponding prefixes flow_x / flow_y, the images with prefix im, cuts off all flow values < -16 and > 16, using method 2 (Brox Optical Flow) on GPU 0, sampling only every 10th frame.
