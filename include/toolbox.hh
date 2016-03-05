#ifndef TOOLBOX_HH
#define TOOLBOX_HH

#include "denseFlow_gpu.hh"

namespace toolbox{

	// Originally included functions
	void drawOptFlowMap(const cv::Mat& flow, cv::Mat& cflowmap, int step, const cv::Scalar& color);
	void convertFlowToImage(const cv::Mat &flow_x, const cv::Mat &flow_y, cv::Mat &img_x, cv::Mat &img_y,
	       double lowerBound, double higherBound);

	#ifdef SERIALIZE_BUFFER
	// Encodes cv::Mat as jpeg and converts to string (and reverse)
	std::string encode( cv::Mat & m );
	cv::Mat decode( std::string & str, bool is_color = false );

	// Serialize/Deserialize a vector of strings to/from file
	void serialize( std::vector<std::string> & vec, std::string filename );
	void deserialize( std::vector<std::string> & vec, std::string filename );

	// Serialize/Deserialize to/from stringstream
	void serializeToString( std::vector<std::string> & vec, std::stringstream & output );
	void deserializeFromString( std::vector<std::string> & vec, std::stringstream & input );

	class Serializer{
		public:
			Serializer( std::string prefix, std::string postfix, int64_t chunk_size ){
				chunk_size_ = chunk_size;
				prefix_ = prefix;
				postfix_ = postfix;
			}

			Serializer( int64_t chunk_size ) : Serializer( "archive", ".ar", chunk_size ){};
			Serializer( std::string pre, std::string post ) : Serializer( pre, post, 1000 ){};
			Serializer( ) : Serializer( "archive", ".ar", 1000 ){};

			inline void push_back( std::string s ){
				data_.push_back( s );
				counter_++;

				if( counter_ >= chunk_size_ )
					sync();

			}

			inline void sync(){
				serialize( data_, prefix_ + "_chk" + boost::lexical_cast<std::string>( chunk_num_ ) + postfix_ );
				counter_ = 0;
				chunk_num_++;
			}

		private:
			std::vector<std::string> data_;
			int64_t chunk_size_ = 0;
			int64_t counter_ = 0;
			int64_t chunk_num_ = 0;

			std::string prefix_ = "";
			std::string postfix_ = "";
	};
	#endif

} // namespace toolbox
#endif