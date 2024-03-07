#include <iostream>
#include <cstdio>
#include <cstring>
#include <random>
#include <chrono>
#include <fstream>

#define ull unsigned int
using namespace std;

int main(int argc , char* argv[]){
  	
  	mt19937 mt(time(nullptr));
  	
  	ull size_of_input = stoi(argv[1]);
	
	auto start_time = chrono::high_resolution_clock::now();
	
	FILE *my_file ;
	my_file = fopen("input.bin", "wb");
	
	for(ull i = 0; i < 2*(ull)stoi(argv[1]) ; ++i){
		
		size_of_input = mt() ;
		//cout<<size_of_input<<endl;
		size_of_input = __builtin_bswap32(size_of_input);
		fwrite(&size_of_input, sizeof(ull), 1, my_file);
	}
	
	fclose(my_file);
    /*
    fstream infile;
    infile.open("input.bin", ios::binary | ios::in);
    infile.seekg(0);
    
    for(ull i = 0; i < 2*(ull)stoi(argv[1]); i++){
      
      ull num;
      ull x = 0;
      for(int j = 0; j < 4; j++){
      
        infile.read((char*)&num , 1);
      	//cout<<num<<" ";
        x = (x << 8) | num;
      }
      cout<<x<<" ";
    }
    cout<<endl;

    infile.close();
	*/
	auto end_time = chrono::high_resolution_clock::now();
	
	double time_taken = (1e-9*(chrono::duration_cast<chrono::nanoseconds>(end_time - start_time)).count());
	
	cout<<"Time Taken: "<<time_taken<<" s"<<endl;
    
	return 0;
}
