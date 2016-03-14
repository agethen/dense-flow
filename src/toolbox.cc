#include "toolbox.hh"

namespace toolbox {

	std::string encode( cv::Mat & m ){
		std::vector<uchar> buf;
		std::vector<int> compression;
		compression.push_back( CV_IMWRITE_JPEG_QUALITY );
		compression.push_back( JPEG_QUALITY );
		cv::imencode(".jpg", m, buf, compression );
		std::string str = std::string( reinterpret_cast<char*>( &buf[0] ), buf.size() );
		return str;
	}

	cv::Mat decode( std::string & str, bool is_color ){
		cv::Mat res;
		std::vector<uchar> buf_in( str.begin(), str.end() );
		if( is_color )
			res = cv::imdecode( buf_in, CV_LOAD_IMAGE_COLOR );
		else
			res = cv::imdecode( buf_in, CV_LOAD_IMAGE_GRAYSCALE );
		return res;
	}

	void serialize( std::vector<std::string> & vec, std::string filename ){
		std::ofstream ofs( filename );
		boost::archive::text_oarchive ar(ofs);
		ar & vec;
	}

	void deserialize( std::vector<std::string> & vec, std::string filename ){
		std::ifstream ifs( filename );
		boost::archive::text_iarchive ar(ifs);
		ar & vec;
	}

	void serializeToString( std::vector<std::string> & vec, std::stringstream & output ){
	  boost::archive::text_oarchive ar( output );
	  ar & vec;
	}

	void deserializeFromString( std::vector<std::string> & vec, std::stringstream & input ){
	  boost::archive::text_iarchive ar( input );
	  ar & vec;
	}


	#ifdef UNUSED
	void drawOptFlowMap(const cv::Mat& flow, cv::Mat& cflowmap, int step, const cv::Scalar& color){
	    for(int y = 0; y < cflowmap.rows; y += step)
	        for(int x = 0; x < cflowmap.cols; x += step)
	        {
	            const cv::Point2f& fxy = flow.at<cv::Point2f>(y, x);
	            cv::line(cflowmap, cv::Point(x,y), cv::Point(cv::saturate_cast<int>(x+fxy.x), cv::saturate_cast<int>(y+fxy.y)),
	                 color);
	            cv::circle(cflowmap, cv::Point(x,y), 2, color, -1);
	        }
	}
	#endif


		void convertFlowToImage(const cv::Mat &flow_x, const cv::Mat &flow_y, cv::Mat &img_x, cv::Mat &img_y,
	       double lowerBound, double higherBound, bool finegrained ) {
		#ifdef PREVIOUS_SCALING
		#define CAST(v, L, H) ((v) > (H) ? 255 : (v) < (L) ? 0 : cvRound(255*((v) - (L))/((H)-(L))))
		for (int i = 0; i < flow_x.rows; ++i) {
			for (int j = 0; j < flow_y.cols; ++j) {
				float x = flow_x.at<float>(i,j);
				float y = flow_y.at<float>(i,j);
				img_x.at<uchar>(i,j) = CAST(x, lowerBound, higherBound);
				img_y.at<uchar>(i,j) = CAST(y, lowerBound, higherBound);
			}
		}
		#undef CAST
		#else

			// If finegrained precision is required, we can save data as RGB jpegs.
			// In that case R represents large movement, G small movement, B should probably be discarded (too fine)
			if( finegrained ){
				cv::Size s = flow_x.size();
				std::vector<cv::Mat> img_x_channels { cv::Mat::zeros( s, CV_8UC1 ), cv::Mat::zeros( s, CV_8UC1 ), cv::Mat::zeros( s, CV_8UC1 ) };
				std::vector<cv::Mat> img_y_channels { cv::Mat::zeros( s, CV_8UC1 ), cv::Mat::zeros( s, CV_8UC1 ), cv::Mat::zeros( s, CV_8UC1 ) };

				// X-Flow

				// Use integer values as is (in 0-255)
				flow_x.convertTo( img_x_channels[2], CV_8UC1, 1, 128 );
				
				//[-1 and 1] --> [0,255]
				cv::Mat tmp = cv::Mat::zeros( s, CV_32F );
				cv::Mat tmp2 = cv::Mat::zeros( s, CV_32F );

				img_x_channels[2].convertTo( tmp, CV_32F, 1, -128 );	// Recast to floats. We now have a rounded version of flow_x
				tmp2 = flow_x - tmp;																	// This gives the digits after the point only
				tmp2.convertTo( img_x_channels[1], CV_8UC1, 128.0, 128 );	// Bring into range for JPEG

				// Y-Flow

				flow_y.convertTo( img_y_channels[2], CV_8UC1, 1, 128 );
				img_y_channels[2].convertTo( tmp, CV_32F, 1, -128 );
				tmp2 = flow_y - tmp;

				tmp2.convertTo( img_y_channels[1], CV_8UC1, 128.0, 128 );

				// Merge matrices
				cv::merge( img_x_channels, img_x );
				cv::merge( img_y_channels, img_y );
			}else{
				// We ignore lowerBound and assume it is lowerBound = -higherBound
				double factor = 256.0/(2.0*higherBound);
				flow_x.convertTo( img_x, CV_8UC1, factor, 128 );	// For simplicity, we assume a range of [-8, 8]. There are (rarely) higher/lower values
				flow_y.convertTo( img_y, CV_8UC1, factor, 128 );
			}
		#endif
	}


}	// namespace toolbox