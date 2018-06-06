all:	jacobi.exe

%.exe:	%.cpp
	nvcc -D_FORCE_INLINES -x cu -I. $< -o $@
