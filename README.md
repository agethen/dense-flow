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

## Compilation
To compile, create 'build' directory, and run 'make'.
