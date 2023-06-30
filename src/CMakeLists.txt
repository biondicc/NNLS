# This demo builds all sources into a library, and then compiles
# and links  main  against the library and all Trilinos libraries.

SET(NNLS_SOURCES NNLS_solver.h NNLS_solver.cpp)
add_custom_target(MyApp SOURCES ${NNLS_SOURCES})



