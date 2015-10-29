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

// Keep how many frames loaded at the same time to compute flow (should be larger than stepping)
#define BUFFER_SIZE 100

// New Image Dimensions
#define DIM_X 340
#define DIM_Y 256

// JPEG Quality
#define JPEG_QUALITY 95

// Don't save frames individually, but rather serialize them to one file
// Activate this flag via -DSERIALIZE_BUFFER
// #define SERIALIZE_BUFFER

void compute_flow( cv::Mat & prev, cv::Mat & cur, std::vector<cv::Mat> & output, int type );
#endif