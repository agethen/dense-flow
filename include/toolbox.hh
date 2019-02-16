#ifndef TOOLBOX_HH
#define TOOLBOX_HH

#include <string>
#include <vector>
#include <memory>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#ifdef SERIALIZE_BUFFER
#include <boost/serialization/vector.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#endif

#include "common.hh"

namespace toolbox{

  inline std::string int_to_string( int64_t val ){
    std::stringstream out;
    out << val;
    return out.str();
  }

  // Originally included functions
  void drawOptFlowMap(const cv::Mat& flow, cv::Mat& cflowmap, int step, const cv::Scalar& color);
  void convertFlowToImage(const cv::Mat &flow_x, const cv::Mat &flow_y, cv::Mat &img_x, cv::Mat &img_y,
         double lowerBound, double higherBound, bool finegrained = false );

  // Encodes cv::Mat as jpeg and converts to string (and reverse)
  std::string encode( const cv::Mat & m );
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

      inline void PushBack( std::string s ){
        data_.push_back( s );
        counter_++;

        if( counter_ >= chunk_size_ )
          sync();

      }

      inline void sync(){
        serialize( data_, prefix_ + "_chk" + int_to_string( chunk_num_ ) + postfix_ );
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

  class IOManager{
    public:
      IOManager(  const std::string img, const std::string flow_x, const std::string flow_y,
                  const std::vector<int64_t> span, const int64_t max_files_chunk, const bool serialize );


      void WriteImg( const cv::Mat & img, const int64_t id );
      void WriteFlow( const cv::Mat & x, const cv::Mat & y, const int64_t id, const int64_t span_id );

      void sync();

    private:
      std::string CreateFilename( const int64_t id, const int64_t span_id, const int64_t type );

      bool serialize_;
      std::string img_;
      std::string flow_x_;
      std::string flow_y_;

      Serializer * archive_i_;
      std::vector< Serializer * > archive_x_;
      std::vector< Serializer * > archive_y_;

  };

} // namespace toolbox
#endif