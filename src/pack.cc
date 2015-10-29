#include "toolbox.hh"

#include <fstream>
#include <string>
#include <vector>
#include <iostream>

// Serialize a collection of JPEGs
// Files to be packed are specified as a list of filenames, separated by newlines

const int archive_size = 10000;

int main( int argc, char ** argv ){

	std::string file = "input.txt";
	std::string out = "out";
	bool is_color = false;

	for( int i = 0; i < argc; i++ ){

		if( std::string( argv[i] ) == "--file" ){
			if( i < argc-1 )
				file = std::string( argv[i+1] );
		}

		if( std::string( argv[i] ) == "--out" ){
			if( i < argc-1 )
				out = std::string( argv[i+1] );
		}

		if( std::string( argv[i] ) == "--color" )
			is_color = true;
	}

	std::vector<std::string> buffer;
	std::ifstream ifs( file );

	std::string filename;

	int chunk = 0;
	int counter = 0;

	while( ifs >> filename ){
		cv::Mat im = cv::imread( filename, (is_color)?CV_LOAD_IMAGE_COLOR:CV_LOAD_IMAGE_GRAYSCALE );

		buffer.push_back( toolbox::encode( im ) );

		if( counter++ > archive_size ){
			toolbox::serialize( buffer, out + "_chk" + boost::lexical_cast<std::string>( chunk++ ) + ".flow" );
			counter = 0;
		}
	}

	toolbox::serialize( buffer, out + "_chk" + boost::lexical_cast<std::string>( chunk++ ) + ".flow" );

	std::cout << "Packed a total of " << (chunk-1)*archive_size+counter << " files in " << chunk << " chunk files." << std::endl;
	return EXIT_SUCCESS;
}