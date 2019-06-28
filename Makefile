
# Makefile

CFLAGS = -I. -O0

lenet: lenet.cu cnnfunc.cu 
	nvcc $(CFLAGS) -o lenet lenet.cu cnnfunc.cu

clean:
	rm -f lenet

.PHONY: clean

