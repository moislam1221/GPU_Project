all:	testUpperTriangle.exe

%.exe:	%.cpp ../UpperTriangular.h
	nvcc -D_FORCE_INLINES -arch=sm_52 -x cu -I. $< -o $@
