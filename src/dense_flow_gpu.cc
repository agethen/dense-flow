#include "dense_flow_gpu.hh"
#include <iostream>
// Compute optical flow between frames t and t+steppings[*]
const std::vector< int64_t > flow_span { 1 };

cv::Ptr<cv::cuda::FarnebackOpticalFlow> alg_farn;
cv::Ptr<cv::cuda::OpticalFlowDual_TVL1> alg_tvl1;
cv::Ptr<cv::cuda::BroxOpticalFlow> alg_brox;

int main(int argc, char** argv){
  bool serialize = false;
  const int64_t max_files_chunk = MAX_FILES_PER_CHUNK;

  #ifdef SERIALIZE_BUFFER
  serialize = true;
  #endif 
  
  // IO operation
  const cv::String keys =
  "{ f vidFile     | | filename of video }"
  "{ i imgFile     | | filename of image component }"
  "{ x xFlowFile   | | filename of flow x component }"
  "{ y yFlowFile   | | filename of flow x component }"
  "{ b bound       | 15 | specify the maximum of optical flow}"
  "{ t type        | 0 | specify the optical flow algorithm }"
  "{ d device_id   | 0 | set gpu id }"
  "{ s step        | 1 | specify the step for frame sampling}"
  "{ o offset      | 0 | specify the offset from where to start}"
  "{ c clip        | 0 | specify maximum length of clip (0=no maximum)}"
  ;

  cv::CommandLineParser cmd(argc, argv, keys);

  std::string vidFile     = cmd.get<std::string>("vidFile");
  std::string imgFile     = cmd.get<std::string>("imgFile");
  std::string xFlowFile   = cmd.get<std::string>("xFlowFile");
  std::string yFlowFile   = cmd.get<std::string>("yFlowFile");
  int bound               = cmd.get<int>("bound");
  int type                = cmd.get<int>("type");
  int device_id           = cmd.get<int>("device_id");
  int step                = cmd.get<int>("step");
  int offset              = cmd.get<int>("offset");
  int len_clip            = cmd.get<int>("clip");

  if( !cmd.check() ){
    cmd.printErrors();
    return EXIT_FAILURE;
  }

  cv::cuda::setDevice( device_id );

  switch( type ){
    case 0:
      alg_farn = cv::cuda::FarnebackOpticalFlow::create();
      break;
    case 1:
      alg_tvl1 = cv::cuda::OpticalFlowDual_TVL1::create();
      break;
    case 2:
      alg_brox = cv::cuda::BroxOpticalFlow::create(0.197f, 50.0f, 0.8f, 10, 77, 10);
      break;
    default:
      alg_brox = cv::cuda::BroxOpticalFlow::create(0.197f, 50.0f, 0.8f, 10, 77, 10);
  }

  Video v( vidFile, step, len_clip );    // Sample every `step`-th frame, up to the `len_clip`-th frame, i.e., floor(len_clip/step) frames total.

  if( !v.is_open() )
    return EXIT_FAILURE;

  if( offset > v.length() ){
    std::cerr << "Offset exceeds length of video." << std::endl;
    return EXIT_FAILURE;
  }

  v.seek( offset );

  // Process video
  toolbox::IOManager io_manager( imgFile, xFlowFile, yFlowFile, flow_span, max_files_chunk, serialize );
  ProcessClip( v, io_manager, type, bound );

  std::cout << "\t .. Finished" << std::endl;
  return EXIT_SUCCESS;
}

void ProcessClip( Video & v, toolbox::IOManager & io_manager, const int type, const int bound ){

  // At each processing step, we require `span+1` frames to compute flow
  auto max_span = std::max_element( flow_span.begin(), flow_span.end() );    

  std::vector< std::pair<int64_t, cv::Mat> > clip;
  v.read( clip, *max_span, true );

  int64_t counter = 0;

  std::cout << "\t" << std::flush;

  while( true ){
    v.read( clip, 1, true );

    if( clip.empty() )
      break;

    if( ++counter % 50 == 0 )
      std::cout << " -- " << counter << std::flush;

    io_manager.WriteImg( clip[0].second, counter );
    
    for( int i = 0; i < flow_span.size(); i++ ){
      int span = flow_span[i];

      if( span >= clip.size() )
        continue;

      cv::Mat flow_x( cv::Size( DIM_X, DIM_Y ),   CV_8UC1 );
      cv::Mat flow_y( cv::Size( DIM_X, DIM_Y ),   CV_8UC1 );

      cv::Mat grey_first, grey_second;
      cv::cvtColor( clip.begin()->second,         grey_first, CV_BGR2GRAY );
      cv::cvtColor( (clip.begin()+span)->second,  grey_second, CV_BGR2GRAY );

      ComputeFlow( grey_first, grey_second, type, bound, flow_x, flow_y );

      io_manager.WriteFlow( flow_x, flow_y, counter, i );
    }

    clip.erase( clip.begin(), clip.begin()+1 );
  }

  std::cout << " -- " << counter << "." << std::endl;

  io_manager.sync();
}

void ComputeFlow( const cv::Mat prev, const cv::Mat cur, const int type, const int bound, cv::Mat & flow_x, cv::Mat & flow_y ){

  // GPU optical flow
  cv::cuda::GpuMat frame_0( prev );
  cv::cuda::GpuMat frame_1( cur );

  cv::cuda::GpuMat d_flow( frame_0.size(), CV_32FC2 );

  switch(type){
  case 0:
    alg_farn->calc(frame_0, frame_1, d_flow );
    break;
  case 1:
    alg_tvl1->calc(frame_0, frame_1, d_flow );
    break;
  case 2:
    cv::cuda::GpuMat d_frame0f, d_frame1f;
    frame_0.convertTo(d_frame0f, CV_32F, 1.0 / 255.0);
    frame_1.convertTo(d_frame1f, CV_32F, 1.0 / 255.0);

    alg_brox->calc(d_frame0f, d_frame1f, d_flow );
    break;
  }

  cv::Mat flow( d_flow );
  cv::Mat tmp_flow[2];

  cv::split( flow, tmp_flow );

  cv::Mat imgX, imgY;
  // We previously rescaled Optical Flow here. Not sure why?
  // cv::resize( tmp_flow[0], imgX, cv::Size( DIM_X, DIM_Y ) );
  // cv::resize( tmp_flow[1], imgY, cv::Size( DIM_X, DIM_Y ) );
  imgX = tmp_flow[0];
  imgY = tmp_flow[1];

	#ifdef TEST_FINEGRAINED
	toolbox::convertFlowToImage( imgX, imgY, flow_x, flow_y, 0, 0, true );
	#else
	toolbox::convertFlowToImage( imgX, imgY, flow_x, flow_y, -bound, bound );
	#endif
}