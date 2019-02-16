#include "toolbox.hh"

namespace toolbox {

	std::string encode( const cv::Mat & m ){
		std::string str = "";

		#ifdef SERIALIZE_BUFFER
		std::vector<uchar> buf;
		std::vector<int> compression;
		compression.push_back( CV_IMWRITE_JPEG_QUALITY );
		compression.push_back( JPEG_QUALITY );
		cv::imencode(".jpg", m, buf, compression );
		str = std::string( reinterpret_cast<char*>( &buf[0] ), buf.size() );
		#endif
		
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

		#ifdef SERIALIZE_BUFFER
		std::ofstream ofs( filename );
		boost::archive::text_oarchive ar(ofs);
		ar & vec;
		#endif

	}

	void deserialize( std::vector<std::string> & vec, std::string filename ){

		#ifdef SERIALIZE_BUFFER
		std::ifstream ifs( filename );
		boost::archive::text_iarchive ar(ifs);
		ar & vec;
		#endif

	}

	void serializeToString( std::vector<std::string> & vec, std::stringstream & output ){

		#ifdef SERIALIZE_BUFFER
	  boost::archive::text_oarchive ar( output );
	  ar & vec;
	  #endif

	}

	void deserializeFromString( std::vector<std::string> & vec, std::stringstream & input ){

		#ifdef SERIALIZE_BUFFER
	  boost::archive::text_iarchive ar( input );
	  ar & vec;
	  #endif

	}


	#ifdef UNUSED
	void drawOptFlowMap(const cv::Mat& flow, cv::Mat& cflowmap, int step, const cv::Scalar& color){
    for(int y = 0; y < cflowmap.rows; y += step){
	    for(int x = 0; x < cflowmap.cols; x += step){
	      const cv::Point2f& fxy = flow.at<cv::Point2f>(y, x);
        cv::line(cflowmap, cv::Point(x,y), cv::Point(cv::saturate_cast<int>(x+fxy.x), cv::saturate_cast<int>(y+fxy.y)),
	                 color);
        cv::circle(cflowmap, cv::Point(x,y), 2, color, -1);
      }
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


	IOManager::IOManager( const std::string img, const std::string flow_x, const std::string flow_y,
												const std::vector<int64_t> span, const int64_t max_files_chunk, const bool serialize ){
		serialize_ = serialize;
		img_ 		= img;
		flow_x_	= flow_x;
		flow_y_ = flow_y;

		archive_i_ = nullptr;
		archive_x_ = std::vector< toolbox::Serializer * >( span.size(), nullptr );
		archive_y_ = std::vector< toolbox::Serializer * >( span.size(), nullptr );

		if( serialize ){
		  if( img_ != "" )
		    archive_i_ = new toolbox::Serializer( img_, ".image", max_files_chunk );

		  if( flow_x_ != "" ){
		    for( int i = 0; i < span.size(); i++ ){
		      std::string tmp_span = (span.size() == 1)?"":"_span" + int_to_string( span[i] );
		      archive_x_[i] = new toolbox::Serializer( flow_x_ + tmp_span, ".flow", max_files_chunk );
		    }
		  }

		  if( flow_y_ != "" ){
		    for( int i = 0; i < span.size(); i++ ){
		      std::string tmp_span = (span.size() == 1)?"":"_span" + int_to_string( span[i] );
		      archive_y_[i] = new toolbox::Serializer( flow_y_ + tmp_span, ".flow", max_files_chunk );
		    }
		  }
		}
	}

	void IOManager::WriteImg( const cv::Mat & img, const int64_t id ){

		if( serialize_ ){
			if( archive_i_ )
				archive_i_->PushBack( toolbox::encode( img ) );
		} else {
			if( img_.size() )
				cv::imwrite( CreateFilename( id, -1, 0 ), img );
      
		}

	}

	void IOManager::WriteFlow( const cv::Mat & x, const cv::Mat & y, const int64_t id, const int64_t span_id ){
		
		if( serialize_ ){
			if( archive_x_[span_id] )
				archive_x_[span_id]->PushBack( toolbox::encode( x ) );

			if( archive_y_[span_id] )
				archive_y_[span_id]->PushBack( toolbox::encode( y ) );
		} else {
			
			if( flow_x_.size() )
				cv::imwrite( CreateFilename( id, span_id, 1 ), x );

			if( flow_y_.size() )
				cv::imwrite( CreateFilename( id, span_id, 2 ), y );

		}

	}

	void IOManager::sync(){

		if( serialize_ ){
			if( archive_i_ )
				archive_i_->sync();

			for( auto p : archive_x_ )
				p->sync();

			for( auto p : archive_y_ )
				p->sync();
		}

	}

	std::string IOManager::CreateFilename( const int64_t id, const int64_t span_id, const int64_t type ){
		std::string name = "";
		std::string span = "";

		// As span == 1 is default setting, we omit span string in that case.
		if( span_id > 1 ){
			span = "_span" + int_to_string( span_id );
		}

		switch( type ){
			case 0:
				name = img_ + int_to_string( id ) + ".jpg";
				break;
			case 1:
				name = flow_x_ + int_to_string( id ) + span + ".jpg";
				break;
			case 2:
				name = flow_y_ + int_to_string( id ) + span + ".jpg";
				break;
			default:
				name = "";
		}
    
		return name;
	}

}	// namespace toolbox