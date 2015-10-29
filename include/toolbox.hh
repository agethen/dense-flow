#ifndef TOOLBOX_HH
#define TOOLBOX_HH

#include "denseFlow_gpu.hh"

namespace toolbox{

	// Encodes cv::Mat as jpeg and converts to string (and reverse)
	std::string encode( cv::Mat & m );
	cv::Mat decode( std::string & str, bool is_color = false );

	// Serialize/Deserialize a vector of strings to/from file
	void serialize( std::vector<std::string> & vec, std::string filename );
	void deserialize( std::vector<std::string> & vec, std::string filename );

	// Serialize/Deserialize to/from stringstream
	void serializeToString( std::vector<std::string> & vec, std::stringstream & output );
	void deserializeFromString( std::vector<std::string> & vec, std::stringstream & input );

	// Originally included functions
	void drawOptFlowMap(const cv::Mat& flow, cv::Mat& cflowmap, int step, const cv::Scalar& color);
	void convertFlowToImage(const cv::Mat &flow_x, const cv::Mat &flow_y, cv::Mat &img_x, cv::Mat &img_y,
	       double lowerBound, double higherBound);

} // namespace toolbox
#endif