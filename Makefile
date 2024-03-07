CC = g++
NCC = nvcc
CFLAGS = -g

all: a4

a4: main.cu
	$(NCC) -Xcompiler -fopenmp -o a4 main.cu
	
clean:
	rm a4
