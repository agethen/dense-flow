include Makefile.config
.PHONY: all
all:
	g++ $(FLAGS) -DSERIALIZE_BUFFER $(INCLUDE) -c src/denseFlow_gpu_cv3.cc
	g++ $(FLAGS) -DSERIALIZE_BUFFER $(INCLUDE) -c src/toolbox.cc
	g++ -o $(OUT)/denseFlow_gpu denseFlow_gpu_cv3.o toolbox.o $(LIB) -lopencv_core -lopencv_cudaoptflow -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -lopencv_videoio -lboost_serialization
	rm *.o
simple:
	g++ $(FLAGS) $(INCLUDE) -c src/denseFlow_gpu_cv3.cc
	g++ $(FLAGS) $(INCLUDE) -c src/toolbox.cc
	g++ -o $(OUT)/denseFlow_gpu denseFlow_gpu_cv3.o toolbox.o $(LIB) -lopencv_core -lopencv_cudaoptflow -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -lopencv_videoio -lboost_serialization
	rm *.o
unpacker:
	g++ $(FLAGS) -DSERIALIZE_BUFFER $(INCLUDE) -c src/unpack.cc
	g++ $(FLAGS) -DSERIALIZE_BUFFER $(INCLUDE) -c src/toolbox.cc
	g++ -o $(OUT)/unpack unpack.o toolbox.o $(LIB) -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lboost_serialization
	rm *.o
repack:
	g++ $(FLAGS) -DSERIALIZE_BUFFER $(INCLUDE) -c src/flow_split.cc
	g++ $(FLAGS) -DSERIALIZE_BUFFER $(INCLUDE) -c src/toolbox.cc
	g++ -o $(OUT)/repack flow_split.o toolbox.o $(LIB) -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lboost_serialization
	rm *.o
packer:
	g++ $(FLAGS) -DSERIALIZE_BUFFER $(INCLUDE) -c src/pack.cc
	g++ $(FLAGS) -DSERIALIZE_BUFFER $(INCLUDE) -c src/toolbox.cc
	g++ -o $(OUT)/pack pack.o toolbox.o $(LIB) -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lboost_serialization
	rm *.o