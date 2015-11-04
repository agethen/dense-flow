#include "denseFlow_gpu.hh"
#include "toolbox.hh"

// Compute optical flow between frames t and t+steppings[*]
const std::vector< unsigned int > steppings { 1 };

// If SERIALIZE_BUFFER, group chunk_size images per file
const int chunk_size = 10000;

cv::Ptr<cv::cuda::FarnebackOpticalFlow> alg_farn;
cv::Ptr<cv::cuda::OpticalFlowDual_TVL1> alg_tvl1;
cv::Ptr<cv::cuda::BroxOpticalFlow> alg_brox;

int main(int argc, char** argv){
	// IO operation
	const cv::String keys =
			"{ f vidFile      | ex2.avi | filename of video }"
			"{ x xFlowFile    | flow_x | filename of flow x component }"
			"{ y yFlowFile    | flow_y | filename of flow x component }"
			"{ i imgFile      | flow_i | filename of flow image}"
			"{ b bound 				| 15 | specify the maximum of optical flow}"
			"{ t type 				| 0 | specify the optical flow algorithm }"
			"{ d device_id    | 0 | set gpu id}"
			"{ s step  				| 1 | specify the step for frame sampling}"
			;

	cv::CommandLineParser cmd(argc, argv, keys);

	std::string vidFile = cmd.get<std::string>("vidFile");
	std::string xFlowFile = cmd.get<std::string>("xFlowFile");
	std::string yFlowFile = cmd.get<std::string>("yFlowFile");
	std::string imgFile = cmd.get<std::string>("imgFile");
	int bound = cmd.get<int>("bound");
  int type  = cmd.get<int>("type");
  int device_id = cmd.get<int>("device_id");
  int step = cmd.get<int>("step");

  if( !cmd.check() ){
  	cmd.printErrors();
  	return 0;
  }

 	std::cout << "Processing file: " << vidFile << std::endl;

	cv::VideoCapture * capture = new cv::VideoCapture(vidFile);
	if(!capture->isOpened()) {
		printf("Could not initialize capturing..\n");
		return -1;
	}

	cv::cuda::setDevice(device_id);

	alg_farn = cv::cuda::FarnebackOpticalFlow::create();
	alg_tvl1 = cv::cuda::OpticalFlowDual_TVL1::create();
	alg_brox = cv::cuda::BroxOpticalFlow::create(0.197f, 50.0f, 0.8f, 10, 77, 10);


	int frame_num = 0;
	int chunk_num = 0;
	
	// We need to keep a backlog of images as we compute flow from t to t+steppings[*]
	std::vector< cv::Mat > frames_buffer;

	// Instead of saving the optical flow as single jpegs, we serialize them into a single file
	std::vector< std::string > output_buffer_image;
	std::vector< std::string > output_buffer_x;
	std::vector< std::string > output_buffer_y;

	while(true) {
		cv::Mat frame;
		*capture >> frame;

		if(frame.empty())
			break;
		
		cv::Mat grey;
		cv::cvtColor( frame, grey, CV_BGR2GRAY);
		frames_buffer.push_back( grey );
				
		if(frame_num == 0) {
			frame_num++;

			int step_t = step;
			while (step_t > 1){
				*capture >> frame;
				step_t--;
			}
			continue;
		}

		while( frames_buffer.size() > BUFFER_SIZE )
			frames_buffer.erase( frames_buffer.begin() );

		std::vector< cv::Mat > tmp_output;

		for( auto d : steppings ){
			if( frames_buffer.size() > d )
				compute_flow( *(frames_buffer.rbegin()+d), *(frames_buffer.rbegin()), tmp_output, type );
			else
				compute_flow( frames_buffer.front(), frames_buffer.back(), tmp_output, type );
		}


		// Save image
		cv::Mat image;
		cv::resize( frame, image, cv::Size( DIM_X, DIM_Y ) );

		#ifndef SERIALIZE_BUFFER
		cv::imwrite( imgFile + "_" + boost::lexical_cast<std::string>( frame_num ) + ".jpg", image );
		#else
		output_buffer_image.push_back( toolbox::encode( image ) );
		#endif

		// Scale & Save flow
		for( unsigned int i = 0; i < steppings.size(); i++ ){
			cv::Mat imgX( tmp_output[2*i].size(), CV_8UC1 );
			cv::Mat imgY( tmp_output[2*i+1].size(), CV_8UC1 );

			toolbox::convertFlowToImage( tmp_output[2*i], tmp_output[2*i+1], imgX, imgY, -bound, bound );
			
			std::string tmp = "_t" + boost::lexical_cast<std::string>( steppings[i] )
											 + "_" + boost::lexical_cast<std::string>( frame_num ) + ".jpg";

			#ifndef SERIALIZE_BUFFER
			cv::imwrite( xFlowFile + tmp, imgX );
			cv::imwrite( yFlowFile + tmp, imgY );
			#else
			output_buffer_x.push_back( toolbox::encode( imgX ) );
			output_buffer_y.push_back( toolbox::encode( imgY ) );
			#endif
		}

		frame_num++;

		int step_t = step;
		while (step_t > 1){
			*capture >> frame;
			step_t--;
		}

		#ifdef SERIALIZE_BUFFER
		if( frame_num % chunk_size == 0 ){
			toolbox::serialize( output_buffer_x, xFlowFile + "_chk" + boost::lexical_cast<std::string>( chunk_num ) + ".flow" );
			toolbox::serialize( output_buffer_y, yFlowFile + "_chk" + boost::lexical_cast<std::string>( chunk_num ) + ".flow" );
			toolbox::serialize( output_buffer_image, imgFile + "_chk" + boost::lexical_cast<std::string>( chunk_num++ ) + ".image" );

			output_buffer_x.clear();
			output_buffer_y.clear();
			output_buffer_image.clear();
		}
		#endif

		if( frame_num % 50 == 0 )
			std::cout << "-- " << frame_num << " " << std::flush;
	}

	#ifdef SERIALIZE_BUFFER
	if( !(output_buffer_x.empty() || output_buffer_y.empty() || output_buffer_image.empty()) ){
		toolbox::serialize( output_buffer_x, xFlowFile + "_chk" + boost::lexical_cast<std::string>( chunk_num ) + ".flow" );
		toolbox::serialize( output_buffer_y, yFlowFile + "_chk" + boost::lexical_cast<std::string>( chunk_num ) + ".flow" );
		toolbox::serialize( output_buffer_image, imgFile + "_chk" + boost::lexical_cast<std::string>( chunk_num++ ) + ".image" );
	}
	#endif

	std::cout << ".. Finished" << std::endl;
	return 0;
}

void compute_flow( cv::Mat & prev, cv::Mat & cur, std::vector<cv::Mat> & output, int type ){

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
		
	output.push_back( imgX );
	output.push_back( imgY );
}
