# Run tests in parallel iff MPI enabled 
IF (MYAPP_MPI)
  set(MPIGO /usr/bin/mpiexec -np 1)
ELSE()
  set(MPIGO  )
ENDIF()

INCLUDE_DIRECTORIES ( ../src
                     ${Trilinos_INCLUDE_DIRS} ${Trilinos_TPL_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
MESSAGE("   Eigen_Dir = ${EIGEN3_INCLUDE_DIR}")
LINK_DIRECTORIES ( ${EIGEN3_INCLUDE_DIR} )
LINK_DIRECTORIES (${Trilinos_LIBRARY_DIRS} ${Trilinos_TPL_LIBRARY_DIRS})

file(GLOB MY_FILES "*.csv")
file(COPY ${MY_FILES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

ADD_EXECUTABLE(Tests.exe test_cases.cpp ../src/NNLS_solver.cpp)
TARGET_LINK_LIBRARIES(Tests.exe ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES}) 
target_link_libraries(Tests.exe Eigen3::Eigen)

ADD_TEST(NAME known_tests
COMMAND ${MPIGO} $<TARGET_FILE:Tests.exe> known)

ADD_TEST(NAME matlab_test
COMMAND ${MPIGO} $<TARGET_FILE:Tests.exe> matlab)

ADD_TEST(NAME no_cols_matrix
COMMAND ${MPIGO} $<TARGET_FILE:Tests.exe> noCols)

ADD_TEST(NAME empty_matrix
COMMAND ${MPIGO} $<TARGET_FILE:Tests.exe> empty)

ADD_TEST(NAME random_problem
COMMAND ${MPIGO} $<TARGET_FILE:Tests.exe> random)

ADD_TEST(NAME zero_RHS
COMMAND ${MPIGO} $<TARGET_FILE:Tests.exe> zeroRHS)

ADD_TEST(NAME dependent_columns
COMMAND ${MPIGO} $<TARGET_FILE:Tests.exe> depCols)

ADD_TEST(NAME wide_matrix
COMMAND ${MPIGO} $<TARGET_FILE:Tests.exe> wide)

ADD_TEST(NAME zero_iter_w_sol
COMMAND ${MPIGO} $<TARGET_FILE:Tests.exe> zeroIter)

ADD_TEST(NAME n_iter
COMMAND ${MPIGO} $<TARGET_FILE:Tests.exe> nIter)

ADD_TEST(NAME max_iter
COMMAND ${MPIGO} $<TARGET_FILE:Tests.exe> maxIter)

