ifndef CUDA_SDK
CUDA_SDK = /opt/NVIDIA_GPU_Computing_SDK
endif
CC = nvcc
CFLAGS = -g -lGL -lGLU -lGLEW -lglut -I$(CUDA_SDK)/C/common/inc -L$(CUDA_SDK)/C/lib -lcutil_i386

all: raymarcher.run

raymarcher.run:
	$(CC) $(CFLAGS) main.cu parseScene.cpp -o raymarcher

clean:
	rm -f *.o *.out

distclean:
	rm -f *.o *.out raymarcher
