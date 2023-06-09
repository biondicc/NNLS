# This demo builds all sources into a library, and then compiles
# and links  main  against the library and all Trilinos libraries.

INCLUDE_DIRECTORIES ( ./  ./src_subdirTBD
                     ${Trilinos_INCLUDE_DIRS} ${Trilinos_TPL_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
MESSAGE("   Eigen_Dir = ${EIGEN3_INCLUDE_DIR}")
LINK_DIRECTORIES ( ${EIGEN3_INCLUDE_DIR} )
LINK_DIRECTORIES (${Trilinos_LIBRARY_DIRS} ${Trilinos_TPL_LIBRARY_DIRS})


ADD_EXECUTABLE(MyApp.exe main_file.cpp)

TARGET_LINK_LIBRARIES(MyApp.exe ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES}) 
target_link_libraries(MyApp.exe Eigen3::Eigen)
