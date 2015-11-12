#ifndef VIDEO_HH
#define VIDEO_HH

#include <cstdint>
#include <denseFlow_gpu.hh>

class Video{
	public:

		Video( std::string filename, int64_t skip ){
			video_ = new cv::VideoCapture( filename );

			if( !video_->isOpened() ){
				std::cerr << "Could not open video " << filename << std::endl;
				video_ = nullptr;
			}

		 	std::cout << "Processing file: " << filename << std::endl;

			parse();
			skip_ = skip;
		}

		Video( std::string filename ) : Video( filename, 0 ){};

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

		// Read all frames
		inline int64_t read( std::vector<cv::Mat> & frames, bool rgb = true ){
			return read( frames, length_frames_, rgb );
		}

		// Read at most num frames
		inline int64_t read( std::vector<cv::Mat> & frames, int64_t num, bool rgb = true ){
			if( !video_ )
				return 0;

			int64_t num_read = 0;

			for( int64_t i = 0; i < num; i++ ){
				cv::Mat image;
				*video_ >> image;
				
				if( image.empty() )
					return num_read;
				
				pos_++;
				num_read++;

				if( !rgb ){
					cv::Mat grey;
					cv::cvtColor( image, grey, CV_BGR2GRAY );
					frames.push_back( grey );
				}else{
					frames.push_back( image );
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
		int64_t fps_ = 0;
		int64_t pos_ = 0;
		int64_t skip_ = 0;
};
#endif