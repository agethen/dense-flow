#include "denseFlow_gpu.hh"
#include "toolbox.hh"
#include "video.hh"

// Compute optical flow between frames t and t+steppings[*]
const std::vector< unsigned int > flow_span { 1 };

cv::Ptr<cv::cuda::FarnebackOpticalFlow> alg_farn;
cv::Ptr<cv::cuda::OpticalFlowDual_TVL1> alg_tvl1;
cv::Ptr<cv::cuda::BroxOpticalFlow> alg_brox;


int main(int argc, char** argv){
	// IO operation
	const cv::String keys =
			"{ f vidFile      | ex2.avi | filename of video }"
			"{ x xFlowFile    | flow_x | filename of flow x component }"
			"{ y yFlowFile    | flow_y | filename of flow x component }"
			"{ b bound 				| 15 | specify the maximum of optical flow}"
			"{ t type 				| 0 | specify the optical flow algorithm }"
			"{ d device_id    | 0 | set gpu id}"
			"{ s step  				| 1 | specify the step for frame sampling}"
			"{ o offset       | 0 | specify the offset from where to start}"
			"{ c clip         | 0 | specify maximum length of clip (0=no maximum)}"
			;

	cv::CommandLineParser cmd(argc, argv, keys);

	std::string vidFile = cmd.get<std::string>("vidFile");
	std::string xFlowFile = cmd.get<std::string>("xFlowFile");
	std::string yFlowFile = cmd.get<std::string>("yFlowFile");
	int bound = cmd.get<int>("bound");
  int type  = cmd.get<int>("type");
  int device_id = cmd.get<int>("device_id");
  int step = cmd.get<int>("step");
  int offset = cmd.get<int>("offset");
  int len_clip = cmd.get<int>("clip");

  if( !cmd.check() ){
  	cmd.printErrors();
  	return 0;
  }

	cv::cuda::setDevice(device_id);

	alg_farn = cv::cuda::FarnebackOpticalFlow::create();
	alg_tvl1 = cv::cuda::OpticalFlowDual_TVL1::create();
	alg_brox = cv::cuda::BroxOpticalFlow::create(0.197f, 50.0f, 0.8f, 10, 77, 10);

	Video v( vidFile, step );

	int length = v.length();

	if( offset > length ){
		std::cerr << "Offset exceeds length of video." << std::endl;
		return EXIT_FAILURE;
	}

	process_clip( v, xFlowFile, yFlowFile, type, bound, offset, len_clip );

	std::cout << "\t .. Finished" << std::endl;
	return EXIT_SUCCESS;
}

void process_clip( Video & v, std::string xFlowFile, std::string yFlowFile, int type, int bound, int offset, int len_clip ){
	// Read a clip
	std::vector<cv::Mat> clip;
	v.seek( offset );
	if( len_clip > 0 )
		int res = v.read( clip, len_clip, false );	// Read a clip of max `len_clip` frames and convert to greyscale
	else
		int res = v.read( clip, false );

	// For each desired span of frames, compute optical flow
	// This allows faster parsing then running the program several times with different values of step
	for( auto span : flow_span ){
		if( span > clip.size() )
			continue;

		#ifdef SERIALIZE_BUFFER
			toolbox::Serializer archive_x( xFlowFile + (flow_span.size()==1?"":"_span" + boost::lexical_cast<std::string>( span )), ".flow", 5000 );
			toolbox::Serializer archive_y( yFlowFile + (flow_span.size()==1?"":"_span" + boost::lexical_cast<std::string>( span )), ".flow", 5000 );
		#endif

		std::cout << "\tProcessing span " << span << std::endl << "\t0" << std::flush;

		int counter = 0;

		for( auto f = clip.begin()+span; f < clip.end(); f++ ){
			cv::Mat flow_x( cv::Size( DIM_X, DIM_Y ), 	CV_8UC1 );
			cv::Mat flow_y( cv::Size( DIM_X, DIM_Y ), 	CV_8UC1 );

			auto f_last = f-span;
			compute_flow( *f_last, *f, flow_x, flow_y, type, bound );

			#ifndef SERIALIZE_BUFFER
			std::string tmp = "_t" + boost::lexical_cast<std::string>( span )
											 + "_" + boost::lexical_cast<std::string>( counter ) + ".jpg";

			cv::imwrite( xFlowFile + tmp, flow_x );
			cv::imwrite( yFlowFile + tmp, flow_y );
			#else
			archive_x.push_back( toolbox::encode( flow_x ) );
			archive_y.push_back( toolbox::encode( flow_y ) );
			#endif

			counter++;
			if( counter % 50 == 0 )
				std::cout << " -- " << counter << std::flush;
		}
		std::cout << " -- " << counter << "." << std::endl;

		#ifdef SERIALIZE_BUFFER
		archive_x.sync();
		archive_y.sync();
		#endif
	}

	clip.clear();
				
}

void compute_flow( cv::Mat & prev, cv::Mat & cur, cv::Mat & flow_x, cv::Mat & flow_y, int type, int bound ){

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
	cv::resize( tmp_flow[0], imgX, cv::Size( DIM_X, DIM_Y ) );
	cv::resize( tmp_flow[1], imgY, cv::Size( DIM_X, DIM_Y ) );
		
	toolbox::convertFlowToImage( imgX, imgY, flow_x, flow_y, -bound, bound );		
}