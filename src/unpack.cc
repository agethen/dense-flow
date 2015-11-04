#include "toolbox.hh"

#include <string>
#include <fstream>
#include <vector>
#include <iostream>

int main( int argc, char ** argv ){

	std::string file = "test/flow_x.flow";
	std::string out = "test_out";
	bool is_color = false;

	for( int i = 0; i < argc; i++ ){
		if( std::string( argv[i] ) == "--file" ){
			if( i < argc-1 )
				file = std::string( argv[i+1] );
		}
		if( std::string( argv[i] ) == "--dir" ){
			if( i < argc-1 )
				out = std::string( argv[i+1] );
		}
		if( std::string( argv[i] ) == "--color" )
			is_color = true;
	}

	std::vector<std::string> data;

	toolbox::deserialize( data, file );
	int c = 0;
	for( auto d : data ){
		cv::Mat im = toolbox::decode( d, is_color );
		cv::imwrite( out + boost::lexical_cast<std::string>( c++ ) + ".jpg", im );
	}
	return 0;
}