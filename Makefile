
# Makefile

CFLAGS = -I.

lenet: lenet.cu cnnfunc.cu cnnfunc_gpu.cu
	nvcc $(CFLAGS) -o lenet lenet.cu cnnfunc.cu cnnfunc_gpu.cu

clean:
	rm -f lenet

.PHONY: clean

