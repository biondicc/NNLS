# CMake generated Testfile for 
# Source directory: /home/calistabiondic/Documents/Prep Work/NNLS/test
# Build directory: /home/calistabiondic/Documents/Prep Work/NNLS/build/test
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(Tests.exe "/usr/bin/mpiexec" "-np" "1" "/home/calistabiondic/Documents/Prep Work/NNLS/build/test/Tests.exe")
set_tests_properties(Tests.exe PROPERTIES  _BACKTRACE_TRIPLES "/home/calistabiondic/Documents/Prep Work/NNLS/test/CMakeLists.txt;19;ADD_TEST;/home/calistabiondic/Documents/Prep Work/NNLS/test/CMakeLists.txt;0;")
