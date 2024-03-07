#include<iostream>
#include<string>
#include<fstream>
#include<algorithm>
#include<vector>
#include<omp.h>
#include<cstring>
#include<cstdio>
#include<chrono>
#include<sys/stat.h>

#define ull unsigned int
using namespace std;

__global__ void radixsort(ull *akey, ull *avalue , ull *skey, ull *svalue, ull size_of_array, ull n){
    extern __shared__ volatile ull array[];
	volatile ull* histogram = array;
    ull tid = threadIdx.x;
    ull offset, ai, bi, exp = 1, cnt = 4;
    int num_of_elements = size_of_array / n;
    int seek_pos;

    if(size_of_array % n >= (tid + 1)){
        num_of_elements++;
        seek_pos = num_of_elements * tid;
    }else{
        seek_pos = (size_of_array % n) + num_of_elements * tid;
    }
    
    while(cnt--){

        for(ull i = 0 ; i < 256; i++){
            histogram[256*tid + i] = 0;
        }
			
        for(ull i = seek_pos ; i < seek_pos + num_of_elements; i++){
            skey[i] = akey[i];
            svalue[i] = avalue[i];
            histogram[256*tid + (skey[i] / exp) % 256]++;
        }
		
        for(ull j = 0; j < 256; j++){
           	
			if(j > 0 && tid == 0){
                histogram[256*tid + j] += histogram[256*(n - 1) + j - 1];  
            }
          
            offset = 1;
			
            for (ull d = n>>1; d > 0; d >>= 1){   
                __syncthreads();
                if (tid < d)    {
                    ai = offset*(2*tid+1)-1;
                    bi = offset*(2*tid+2)-1;
                    histogram[256*bi + j] += histogram[256*ai + j];
            
				}
                offset *= 2;
            }

            offset >>= 1;

            for (ull d = 1; d < (n >> 1); d = 2*d + 1){        

                offset >>= 1;
                __syncthreads();

                if (tid < d)      {
					
                    ai = 2*offset*(tid + 1) - 1;
                    bi = offset*(2*(tid + 1) + 1) - 1;
                    histogram[256*bi + j] += histogram[256*ai + j];
                }

            }
            __syncthreads();
        }
		
        for(int i = seek_pos + num_of_elements - 1 ; i >= seek_pos ; i--){
			__syncthreads();    
            akey[histogram[256*tid + (skey[i] / exp)% 256] - 1] = skey[i];
            avalue[histogram[256*tid + (skey[i] / exp)% 256] - 1] = svalue[i];
            histogram[256*tid + (skey[i] / exp)% 256]--;
        }
		
        exp *= 256;
        __syncthreads();
    }
}

void sortongpu(ull *akey, ull *avalue, ull size_of_array){

    ull *d_akey, *d_avalue, *skey, *svalue ;

    cudaMalloc((void**)&d_akey, size_of_array*sizeof(ull));
    cudaMalloc((void**)&d_avalue, size_of_array*sizeof(ull));
	cudaMalloc((void**)&skey, size_of_array*sizeof(ull));
	cudaMalloc((void**)&svalue, size_of_array*sizeof(ull));

    cudaMemcpy(d_akey, akey, size_of_array*sizeof(ull), cudaMemcpyHostToDevice);
    cudaMemcpy(d_avalue, avalue, size_of_array*sizeof(ull), cudaMemcpyHostToDevice);

    ull num_of_threads = 1;
    while(size_of_array > (num_of_threads << 1)){
        num_of_threads = num_of_threads << 1;
    }
   
	num_of_threads = min(32, num_of_threads);
    ull sharedmemory = (num_of_threads*256)*sizeof(ull);

    radixsort<<<1,num_of_threads, sharedmemory>>>(d_akey, d_avalue, skey, svalue, size_of_array , num_of_threads);
	
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();

    cudaMemcpy(akey, d_akey, size_of_array*sizeof(ull), cudaMemcpyDeviceToHost);
    cudaMemcpy(avalue, d_avalue, size_of_array*sizeof(ull), cudaMemcpyDeviceToHost);

    error = cudaGetLastError();
    if(error != cudaSuccess){
        cout<<"Cuda Error: 2 "<<cudaGetErrorString(error)<<endl;
        exit(-1);
    }
	cudaFree(skey);
	cudaFree(svalue);
    cudaFree(d_akey);
    cudaFree(d_avalue);
}

void update_ele_cnt(ull *realsplit , int n, ull *elecnt , ull *key, ull l, ull r){

	for(int i = 0 ; i < n ; i++){
		
		ull idx = lower_bound(key + l, key + r + 1, realsplit[i] + 1) - key;
		#pragma omp atomic  
			elecnt[i] += idx - l;		
	}
}

void copydata(ull *key_copy, ull* value_copy, ull *key, ull *value, ull l, ull r){

	for(ull i = l ; i <= r; i++){
		key[i] = key_copy[i];
        value[i] = value_copy[i];
	}
	return;
}

int main(int argc, char *argv[]){
	
	string temp = argv[3];
	int p = stoi(temp.substr(4, temp.size() - 4 + 1));

    string inputfile = argv[1];
	inputfile = inputfile.substr(12, inputfile.size() - 12 + 1);
		
	struct stat results;
	ull size_of_file;
	if(stat(inputfile.c_str(), &results) == 0){
		size_of_file = results.st_size;
    	}
	ull size_of_input = size_of_file/8;

    ull*key = new ull[size_of_input];
    ull*value = new ull[size_of_input];
    
    #pragma omp parallel for num_threads(p)
    for(int i = 0; i < p; i++){
        fstream infile;
		infile.open(inputfile.c_str(), ios::binary | ios::in);

        ull seek_pos = 0;
		ull num_of_elements = size_of_input/p;
		ull rank = omp_get_thread_num();
		
		if(size_of_input % p >= (rank + 1)){
			num_of_elements++;
			seek_pos = num_of_elements * rank;
		}else{
			seek_pos = (size_of_input % p) + num_of_elements * rank;
		}

        infile.seekg(8*seek_pos);

		for(ull i = 0; i < num_of_elements ; i++){
		    for(int k = 0; k < 2; k++){
                ull num = 0;
                ull x = 0;
                for(int j = 0; j < 4; j++){
                    infile.read((char*)&num , 1);
                    x = (x << 8) | num;
                }
                if(k == 0){
                    key[seek_pos + i] = x;
                }else{
                    value[seek_pos + i]= x;
                }	  
		    }
		}

        infile.close();

    }

    ull *adj = new ull[p+1];
	adj[0] = 0;
	for(ull i= 1; i <= (ull)p; i++){
		if(i-1 < size_of_input % p){
			adj[i] = (size_of_input/p) + 1;
		}else{
			adj[i] = (size_of_input/p);
		}
		adj[i] += adj[i-1];
	}
    ull *pseudosplit = new ull[p*p];

    #pragma omp parallel for num_threads(p)
    for(ull i = 1; i <= (ull)p; i++ ){
		
        ull* akey = new ull[adj[i] - adj[i - 1]];
        ull* avalue = new ull[adj[i] - adj[i - 1]];

        for(ull k = adj[i-1], l = 0; k <= adj[i] - 1; k++, l++){
            akey[l] = key[k];
            avalue[l] = value[k];
        }
		
        sortongpu(akey, avalue, adj[i] - adj[i - 1]);
		
        for(ull k = adj[i-1], l = 0; k <= adj[i] - 1; k++, l++){
            key[k] = akey[l];
            value[k] = avalue[l];
        }
      	 
        ull tt = 0;
        for(ull k = adj[i-1]; k <= adj[i] - 1; k += (size_of_input/p)/p){
            
            pseudosplit[(i-1)*p + tt] = key[k];
            
            tt++;
            if(tt % p == 0){
                break;
            }
        }		
		
        delete[] akey;
        delete[] avalue;
    }
		
    stable_sort(pseudosplit, pseudosplit + p*p );

	ull *realsplit = new ull[p-1];
	for(int i = 0; i < p - 1; i++){
		realsplit[i] = pseudosplit[(i+1)*p];
	} 

	ull *elecnt = new ull[p];
	for(int i = 0; i < p ; i++){
		elecnt[i] = 0;
	}
	
	#pragma omp parallel for num_threads(p)
	for(ull i = 1; i <= (ull)p ; i++){
		update_ele_cnt(realsplit, p - 1, elecnt, key , adj[i-1] , adj[i] - 1);
	}

	elecnt[p-1] = size_of_input;
	for(int i = p - 1; i >= 1 ; i--){
		elecnt[i] -= elecnt[i-1];
	}	
	
	ull* pos = new ull[p+1];
	pos[0]= 0;
	for(int i = 1; i < p+1; i++){
		pos[i] = elecnt[i-1] + pos[i-1];
	}

	ull** start_pos = new ull*[p];
	for(int i = 0; i < p; i++){
		start_pos[i] = new ull[p];
	}

	#pragma omp parallel num_threads(p)
	{
		int curr_bucket = omp_get_thread_num();
		ull l = adj[curr_bucket], r = adj[curr_bucket + 1] - 1; 
		for(int i = 0; i < p-1; i++){
			ull idx = lower_bound(key + l, key + r + 1, realsplit[i] + 1) - key;
			start_pos[curr_bucket][i] = idx - l;
		}
		start_pos[curr_bucket][p-1] = r - l + 1;
	}
	
	ull *key_copy = new ull[size_of_input];
    ull *value_copy = new ull[size_of_input];

	#pragma omp parallel num_threads(p)
	{
		
		ull to_bucket = omp_get_thread_num();
		
		ull base_copy = pos[to_bucket] , from_bucket = 0;
		
		while(from_bucket < (ull) p){
			ull offset_data = 0, till_pos = start_pos[from_bucket][to_bucket], base_data = adj[from_bucket];
			
			if(to_bucket != 0){
				offset_data = start_pos[from_bucket][to_bucket - 1];
			}
			
			for(ull i = offset_data ; i < till_pos; i++, base_copy++){
				key_copy[base_copy] = key[base_data + i];
                value_copy[base_copy] = value[base_data + i];	
			}
			
			from_bucket++;
		}
		
	}

    #pragma omp parallel for num_threads(p)
	for(int i = 1; i <= p; i++){
		copydata(key_copy, value_copy, key, value,adj[i-1],adj[i]-1);
	}

    #pragma omp parallel for num_threads(p)
	for(int i = 1; i <= p; i++){

        ull* akey = new ull[pos[i] - pos[i - 1]];
        ull* avalue = new ull[pos[i] - pos[i - 1]];

        for(ull k = pos[i-1], l = 0; k <= pos[i] - 1; k++, l++){
            akey[l] = key[k];
            avalue[l] = value[k];
        }

        sortongpu(akey, avalue, pos[i] - pos[i - 1]);

        for(ull k = pos[i-1], l = 0; k <= pos[i] - 1; k++, l++){
            key[k] = akey[l];
            value[k] = avalue[l];
        }
	}
		
    string outputfile = argv[2];
	outputfile = outputfile.substr(13, outputfile.size() - 13 + 1);
       
    FILE *my_file ;
    my_file = fopen(outputfile.c_str(), "w+");
    
    for(ull i = 0; i < size_of_input ; ++i){

        key[i] = __builtin_bswap32(key[i]);
        fwrite(&key[i], sizeof(ull), 1, my_file);
        value[i] = __builtin_bswap32(value[i]);
        fwrite(&value[i], sizeof(ull), 1, my_file);
    }
    
    fclose(my_file);

    delete[] pseudosplit;
	delete[] adj;
	delete[] realsplit;
	delete[] elecnt;
	for(int i = 0; i < p; i++){
		delete start_pos[i];
	}
	delete[] start_pos;
	delete[] key_copy;
    delete[] value_copy;
		
    return 0;
}
   
