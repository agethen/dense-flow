#ifndef VIDEO_HH
#define VIDEO_HH

#include <cstdint>

#include <vector>
#include <string>
#include <iostream>
#include <utility>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "common.hh"

class Video{
	public:

		Video( std::string filename, int64_t skip, int64_t len ){
			if( filename == "" ){
				std::cerr << "No video file specified." << std::endl;
				video_ = nullptr;
				return;
			}

			video_ = new cv::VideoCapture( filename );

			if( !video_->isOpened() ){
				std::cerr << "Could not open video " << filename << std::endl;
				video_ = nullptr;
			}

			parse();
			skip_ = skip;
			max_length_ = len;
		 	
		 	std::cout << "Processing file: " << filename << " (" << length_frames_ << " frames)" << std::endl;
		}

		Video( std::string filename ) : Video( filename, 0, 0 ){};

		inline bool is_open(){
			return video_ != nullptr;
		}

		inline int64_t real_length(){
			if( !video_ )
				return 0;

			int64_t count = 0;

			seek( 0 );
			cv::Mat dummy;
			while( true ){
				*video_ >> dummy;

				if( dummy.empty() )
					break;

				count++;
			}
			seek( 0 );
			length_frames_ = count;
			return count;
		}
		
		inline int64_t length(){
			if( !video_ )
				return 0;
			return length_frames_;
		}

		inline void seek( int64_t pos ){
			if( !video_ )
				return;

			video_->set( CV_CAP_PROP_POS_FRAMES, pos );
			pos_ = pos;
		}

		inline int64_t pos(){
			return pos_;
		}

		inline int64_t fps(){
			return fps_;
		}

		// Read all frames (until max)
		inline int64_t read( std::vector< std::pair< int64_t, cv::Mat> > & frames, bool rgb = true ){
			return read( frames, length_frames_, rgb );
		}

		// Read at most num frames
		inline int64_t read( std::vector< std::pair< int64_t, cv::Mat> > & frames, int64_t num, bool rgb = true ){
			if( !video_ )
				return 0;

			int64_t num_read = 0;

			for( int64_t i = 0; i < num; i++ ){

				if( max_frame_ > 0 && pos_ >= max_frame_ )
					return num_read;

				cv::Mat image;
				*video_ >> image;
				
				if( image.empty() )
					return num_read;
				
				pos_++;
				num_read++;

				if( !rgb ){
					cv::Mat grey;
					cv::cvtColor( image, grey, CV_BGR2GRAY );
					frames.push_back( std::make_pair( pos_, grey ) );
				}else{
					frames.push_back( std::make_pair( pos_, image ) );
				}

				for( int64_t j = 1; j < skip_; j++ ){
					*video_ >> image;

					if( image.empty() )
						return num_read;

					pos_++;
				}
			}

			return num_read;
		}

	private:
		inline void parse(){
			if( !video_ )
				return;

			length_frames_ = video_->get( CV_CAP_PROP_FRAME_COUNT );
			fps_ = video_->get( CV_CAP_PROP_FPS );
		}

		cv::VideoCapture * video_ = nullptr;
		int64_t length_frames_ = 0;
		int64_t max_frame_ = 0;
		int64_t max_length_ = 0;
		int64_t fps_ = 0;
		int64_t pos_ = 0;
		int64_t skip_ = 0;
};
#endif