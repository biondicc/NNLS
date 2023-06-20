# CMAKE File for "MyApp" application building against an installed Trilinos

cmake_minimum_required(VERSION 2.7)

# Use Trilinos_PREFIX, if the user set it, to help find Trilinos.
# The final location will actually be held in Trilinos_DIR which must
# point at "<prefix>/lib/cmake/Trilinos", but this helps the search.
SET(CMAKE_PREFIX_PATH ${Trilinos_PREFIX} ${CMAKE_PREFIX_PATH})

# Get Trilinos as one entity
FIND_PACKAGE(Trilinos REQUIRED)
FIND_PACKAGE(Eigen3 3.3 REQUIRED)


# Make sure to use same compilers and flags as Trilinos
SET(CMAKE_CXX_COMPILER ${Trilinos_CXX_COMPILER} )
SET(CMAKE_C_COMPILER ${Trilinos_C_COMPILER} )
SET(CMAKE_Fortran_COMPILER ${Trilinos_Fortran_COMPILER} )

SET(CMAKE_CXX_FLAGS  "${Trilinos_CXX_COMPILER_FLAGS} ${CMAKE_CXX_FLAGS}")
SET(CMAKE_C_FLAGS  "${Trilinos_C_COMPILER_FLAGS} ${CMAKE_C_FLAGS}")
SET(CMAKE_Fortran_FLAGS  "${Trilinos_Fortran_COMPILER_FLAGS} ${CMAKE_Fortran_FLAGS}")
#
# End of setup and error checking
#  NOTE: PROJECT command checks for compilers, so this statement
#        is moved AFTER setting CMAKE_CXX_COMPILER from Trilinos

PROJECT(MyApp)

# MPI check 
MESSAGE("-- Checking if MPI is enabled in Trilinos:")
LIST(FIND Trilinos_TPL_LIST MPI MPI_List_ID)
IF (MPI_List_ID GREATER -1)
  MESSAGE("-- Checking if MPI is enabled in Trilinos: MPI ENABLED")
  SET(MYAPP_MPI TRUE)
  ADD_DEFINITIONS(-DMYAPP_MPI)
ELSE()
  MESSAGE("-- Checking if MPI is enabled in Trilinos: MPI NOT ENABLED")
  SET(MYAPP_MPI FALSE)
ENDIF()

# Set optional dependency in MyApp on Epetra package: 
#   this toggles code within  #ifdef MYAPP_EPETRA 
MESSAGE("-- Looking for Epetra:")
LIST(FIND Trilinos_PACKAGE_LIST Epetra Epetra_List_ID)
IF (Epetra_List_ID GREATER -1)
  ADD_DEFINITIONS(-DMYAPP_EPETRA)
  MESSAGE("-- Looking for Epetra: -- found, compiling with -DMYAPP_EPETRA")
  SET(MYAPP_EPETRA TRUE)
ELSE()
  MESSAGE("-- Looking for Epetra: -- not found.")
  SET(MYAPP_EPETRA FALSE)
ENDIF()

if (TARGET Eigen3::Eigen)
  message("Eigen was found")
endif (TARGET Eigen3::Eigen) 

# Compile source code
ADD_SUBDIRECTORY(src)

#Turn on test to be run by ctest
ENABLE_TESTING()
ADD_SUBDIRECTORY(test)
