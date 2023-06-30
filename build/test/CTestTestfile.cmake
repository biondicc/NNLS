# CMake generated Testfile for 
# Source directory: /home/calistabiondic/Documents/Prep Work/NNLS/test
# Build directory: /home/calistabiondic/Documents/Prep Work/NNLS/build/test
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(known_tests "/usr/bin/mpiexec" "-np" "1" "/home/calistabiondic/Documents/Prep Work/NNLS/build/test/Tests.exe" "known")
set_tests_properties(known_tests PROPERTIES  _BACKTRACE_TRIPLES "/home/calistabiondic/Documents/Prep Work/NNLS/test/CMakeLists.txt;22;ADD_TEST;/home/calistabiondic/Documents/Prep Work/NNLS/test/CMakeLists.txt;0;")
add_test(matlab_test "/usr/bin/mpiexec" "-np" "1" "/home/calistabiondic/Documents/Prep Work/NNLS/build/test/Tests.exe" "matlab")
set_tests_properties(matlab_test PROPERTIES  _BACKTRACE_TRIPLES "/home/calistabiondic/Documents/Prep Work/NNLS/test/CMakeLists.txt;25;ADD_TEST;/home/calistabiondic/Documents/Prep Work/NNLS/test/CMakeLists.txt;0;")
