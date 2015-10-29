include Makefile.config
.PHONY: all
all:
	g++ $(FLAGS) $(INCLUDE) -c src/denseFlow_gpu_cv3.cc
	g++ $(FLAGS) $(INCLUDE) -c src/toolbox.cc
	g++ -o build/denseFlow_gpu denseFlow_gpu_cv3.o toolbox.o $(LIB) -l:libopencv_core.so.3.0 -l:libopencv_cudaoptflow.so.3.0 -l:libopencv_imgcodecs.so.3.0 -l:libopencv_highgui.so.3.0 -l:libopencv_imgproc.so.3.0 -l:libopencv_videoio.so.3.0 -lboost_serialization
	rm *.o
serialize:
	g++ $(FLAGS) -DSERIALIZE_BUFFER $(INCLUDE) -c src/denseFlow_gpu_cv3.cc
	g++ $(FLAGS) -DSERIALIZE_BUFFER $(INCLUDE) -c src/toolbox.cc
	g++ -o build/denseFlow_gpu denseFlow_gpu_cv3.o toolbox.o $(LIB) -l:libopencv_core.so.3.0 -l:libopencv_cudaoptflow.so.3.0 -l:libopencv_imgcodecs.so.3.0 -l:libopencv_highgui.so.3.0 -l:libopencv_imgproc.so.3.0 -l:libopencv_videoio.so.3.0 -lboost_serialization
	rm *.o
unpacker:
	g++ $(FLAGS) $(INCLUDE) -c src/unpack.cc
	g++ $(FLAGS) $(INCLUDE) -c src/toolbox.cc
	g++ -o build/unpack unpack.o toolbox.o $(LIB) -l:libopencv_core.so.3.0 -l:libopencv_imgcodecs.so.3.0 -l:libopencv_highgui.so.3.0 -lboost_serialization
	rm *.o
packer:
	g++ $(FLAGS) $(INCLUDE) -c src/pack.cc
	g++ $(FLAGS) $(INCLUDE) -c src/toolbox.cc
	g++ -o build/pack pack.o toolbox.o $(LIB) -l:libopencv_core.so.3.0 -l:libopencv_imgcodecs.so.3.0 -l:libopencv_highgui.so.3.0 -lboost_serialization
	rm *.o