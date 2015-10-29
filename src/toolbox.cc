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
	       double lowerBound, double higherBound) {
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
			double factor = 256.0/higherBound;
			flow_x.convertTo( img_x, CV_8UC1, factor, 128 );	// For simplicity, we assume a range of [-8, 8]. There are (rarely) higher/lower values
			flow_y.convertTo( img_y, CV_8UC1, factor, 128 );
		#endif
	}

}	// namespace toolbox