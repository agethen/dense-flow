#ifndef DENSEFLOW_GPU_HH
#define DENSEFLOW_GPU_HH

#include <fstream>
#include <cstdio>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>

#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaarithm.hpp>

#include "toolbox.hh"
#include "video.hh"

#include "common.hh"

void ProcessClip( Video & v, toolbox::IOManager & io_manager, const int type, const int bound );
void ComputeFlow( const cv::Mat prev, const cv::Mat cur, const int type, const int bound, cv::Mat & flow_x, cv::Mat & flow_y );

#endif