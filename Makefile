
# Makefile

CFLAGS = -I. --generate-code arch=compute_52,code=sm_52 -use_fast_math

lenet: lenet.cu cnnfunc.cu cnnfunc_gpu.cu
	nvcc $(CFLAGS) -o lenet lenet.cu cnnfunc.cu cnnfunc_gpu.cu

clean:
	rm -f lenet

.PHONY: clean

