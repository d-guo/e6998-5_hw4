SDK_INSTALL_PATH :=  /usr
NVCC=$(SDK_INSTALL_PATH)/local/cuda/bin/nvcc
LIB       :=  -L$(SDK_INSTALL_PATH)/lib64 -L$(SDK_INSTALL_PATH)/samples/common/lib/linux/x86_64
#INCLUDES  :=  -I$(SDK_INSTALL_PATH)/include -I$(SDK_INSTALL_PATH)/samples/common/inc
OPTIONS   :=  -O3
#--maxrregcount=100 --ptxas-options -v 

TAR_FILE_NAME  := YourNameCUDA1.tar
EXECS :=  conv conv_tiled conv_cudnn
all:$(EXECS)

#######################################################################
clean:
	rm -f $(EXECS) *.o

#######################################################################
tar:
	tar -cvf $(TAR_FILE_NAME) Makefile *.h *.cu *.pdf *.txt
#######################################################################

timer.o : timer.cu timer.h
	${NVCC} $< -c -o $@ $(OPTIONS)

#######################################################################
convKernel00.o : convKernel00.cu
	${NVCC} $< -c -o $@ $(OPTIONS)

conv : conv.cu convKernel.h convKernel00.o timer.o
	${NVCC} $< convKernel00.o -o $@ $(LIB) timer.o $(OPTIONS)

convKernel01.o : convKernel01.cu
	${NVCC} $< -c -o $@ $(OPTIONS)

conv_tiled : conv_tiled.cu convKernel.h convKernel01.o timer.o
	${NVCC} $< convKernel01.o -o $@ $(LIB) timer.o $(OPTIONS)

conv_cudnn : conv_cudnn.cu timer.o
	${NVCC} $< -o $@ $(LIB) timer.o $(OPTIONS) -lcudnn
