#ifndef DENSEFLOW_GPU_HH
#define DENSEFLOW_GPU_HH

#include "opencv2/core/core.hpp"
#include "opencv2/core/utility.hpp"

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "opencv2/cudaoptflow.hpp"
#include "opencv2/cudaarithm.hpp"

#include <boost/lexical_cast.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <fstream>
#include <stdio.h>
#include <iostream>
#include "video.hh"

// Keep how many frames loaded at the same time to compute flow (should be larger than stepping)
#define BUFFER_SIZE 10

// New Image Dimensions
#define DIM_X 340
#define DIM_Y 256

// JPEG Quality
#define JPEG_QUALITY 85

#ifdef SERIALIZE_BUFFER
    #define MAX_FILES_PER_CHUNK 10000
#endif

// Don't save frames individually, but rather serialize them to one file
// Activate this flag via -DSERIALIZE_BUFFER
// #define SERIALIZE_BUFFER

void process_clip( Video & v, std::string imgFile, std::string xFlowFile, std::string yFlowFile, int type, int bound );
void compute_flow( cv::Mat prev, cv::Mat cur, cv::Mat & flow_x, cv::Mat & flow_y, int type, int bound );
#endif