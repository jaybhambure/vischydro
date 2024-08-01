# MPI
Petsc is compiled with MPI. The generic name is mpicxx. on my
machine it is something longer e.g. mpic++-mpich-clang17.
Check that you can run an mpi program. On my machine this is mpiexec-mpich-clang17
```
mpicxx -o hworld.exe hworld.cxx 
mpiexec -n 2 ./hworld.exe
```

# pkg-config
If this works next check that petsc is installed correctly. 
pkg-config is the way to go. Check that this produces
the compile flags for petsc:
```
pkg-config --cflags petsc
pkg-config --libs petsc 
```

# Example 1
```
mpicxx -o petsc1.exe `pkg-config --cflags` petsc1.cpp `pkg-config --libs`
mpiexec  ./petsc1.exe
```
# Example 2 with hdf5
```
mpicxx  -o petsc2.exe `pkg-config --cflags` petsc2.cpp `pkg-config --libs`
mpiexec  ./petsc2.exe
```
# The vischydro code
```
mpicxx -o vischydro `pkg-config --cflags` vischydro.cpp jsoncpp.cpp  `pkg-config --libs`
python vischydro_example.py 
```

