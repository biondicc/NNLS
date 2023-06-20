# Run tests in parallel iff MPI enabled 
IF (MYAPP_MPI)
  set(MPIGO /usr/bin/mpiexec -np 1)
ELSE()
  set(MPIGO  )
ENDIF()

INCLUDE_DIRECTORIES ( ./  ./src_subdirTBD
                     ${Trilinos_INCLUDE_DIRS} ${Trilinos_TPL_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
MESSAGE("   Eigen_Dir = ${EIGEN3_INCLUDE_DIR}")
LINK_DIRECTORIES ( ${EIGEN3_INCLUDE_DIR} )
LINK_DIRECTORIES (${Trilinos_LIBRARY_DIRS} ${Trilinos_TPL_LIBRARY_DIRS})

ADD_EXECUTABLE(Tests.exe known_test_cases.cpp)
TARGET_LINK_LIBRARIES(Tests.exe ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES}) 
target_link_libraries(Tests.exe Eigen3::Eigen)
ADD_TEST(NAME Tests.exe
COMMAND ${MPIGO} $<TARGET_FILE:Tests.exe>)


