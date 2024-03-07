#include<iostream>
#include<fstream>
#include<sys/stat.h>
#define ull unsigned int

using namespace std;

int main(){
	
    fstream infile;
		infile.open("output.bin", ios::binary | ios::in);
		infile.seekg(0);
		bool check = true;
		ull prev = 0, curr = 0;
		
		struct stat results;
		ull size_of_file;
    if(stat("output.bin", &results) == 0){
      size_of_file = results.st_size;
    }
    ull size_of_input = size_of_file/8; 
      
    for(ull i = 0; i < size_of_input ; i++){
		for(int k = 0; k < 2; k++){
			ull num = 0;
       		ull x = 0;
        	for(int j = 0; j < 4; j++){
          		infile.read((char*)&num , 1);
          		x = (x << 8) | num;
        	}
			if(k == 0){
				curr = x;
			}
		}
        
        if(prev > curr){
          	check = false;
        }
        prev = curr;
    }
    infile.close();
    
    if(check){
      cout<<"Array is Sorted"<<endl;
    }else{
      cout<<"Array is Not Sorted"<<endl;
    }
      
	return 0;
}