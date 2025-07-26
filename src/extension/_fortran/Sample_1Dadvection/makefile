FC = gfortran
FLAGS = -O2 -fopenmp
OBJS = Sample_1D_advection.o Sample_1D_advection_main.o

advect_test: $(OBJS)
	$(FC) $(FLAGS) -o $@ $(OBJS)

%.o: %.f90
	$(FC) $(FLAGS) -c $<

clean:
	rm -f *.o *.mod advect_test
