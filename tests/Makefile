tests = testUpperTriangleCopy.exe testUpperTriangleIncrement.exe

all:	${tests}

%.exe:	%.cpp ../UpperTriangular.h
	nvcc -D_FORCE_INLINES -arch=sm_52 -x cu -I. $< -o $@

test:	${tests}
	./testUpperTriangleCopy.exe
	./testUpperTriangleIncrement.exe
