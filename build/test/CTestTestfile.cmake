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
add_test(no_cols_matrix "/usr/bin/mpiexec" "-np" "1" "/home/calistabiondic/Documents/Prep Work/NNLS/build/test/Tests.exe" "noCols")
set_tests_properties(no_cols_matrix PROPERTIES  _BACKTRACE_TRIPLES "/home/calistabiondic/Documents/Prep Work/NNLS/test/CMakeLists.txt;28;ADD_TEST;/home/calistabiondic/Documents/Prep Work/NNLS/test/CMakeLists.txt;0;")
add_test(empty_matrix "/usr/bin/mpiexec" "-np" "1" "/home/calistabiondic/Documents/Prep Work/NNLS/build/test/Tests.exe" "empty")
set_tests_properties(empty_matrix PROPERTIES  _BACKTRACE_TRIPLES "/home/calistabiondic/Documents/Prep Work/NNLS/test/CMakeLists.txt;31;ADD_TEST;/home/calistabiondic/Documents/Prep Work/NNLS/test/CMakeLists.txt;0;")
add_test(random_problem "/usr/bin/mpiexec" "-np" "1" "/home/calistabiondic/Documents/Prep Work/NNLS/build/test/Tests.exe" "random")
set_tests_properties(random_problem PROPERTIES  _BACKTRACE_TRIPLES "/home/calistabiondic/Documents/Prep Work/NNLS/test/CMakeLists.txt;34;ADD_TEST;/home/calistabiondic/Documents/Prep Work/NNLS/test/CMakeLists.txt;0;")
add_test(zero_RHS "/usr/bin/mpiexec" "-np" "1" "/home/calistabiondic/Documents/Prep Work/NNLS/build/test/Tests.exe" "zeroRHS")
set_tests_properties(zero_RHS PROPERTIES  _BACKTRACE_TRIPLES "/home/calistabiondic/Documents/Prep Work/NNLS/test/CMakeLists.txt;37;ADD_TEST;/home/calistabiondic/Documents/Prep Work/NNLS/test/CMakeLists.txt;0;")
add_test(dependent_columns "/usr/bin/mpiexec" "-np" "1" "/home/calistabiondic/Documents/Prep Work/NNLS/build/test/Tests.exe" "depCols")
set_tests_properties(dependent_columns PROPERTIES  _BACKTRACE_TRIPLES "/home/calistabiondic/Documents/Prep Work/NNLS/test/CMakeLists.txt;40;ADD_TEST;/home/calistabiondic/Documents/Prep Work/NNLS/test/CMakeLists.txt;0;")
add_test(wide_matrix "/usr/bin/mpiexec" "-np" "1" "/home/calistabiondic/Documents/Prep Work/NNLS/build/test/Tests.exe" "wide")
set_tests_properties(wide_matrix PROPERTIES  _BACKTRACE_TRIPLES "/home/calistabiondic/Documents/Prep Work/NNLS/test/CMakeLists.txt;43;ADD_TEST;/home/calistabiondic/Documents/Prep Work/NNLS/test/CMakeLists.txt;0;")
add_test(zero_iter_w_sol "/usr/bin/mpiexec" "-np" "1" "/home/calistabiondic/Documents/Prep Work/NNLS/build/test/Tests.exe" "zeroIter")
set_tests_properties(zero_iter_w_sol PROPERTIES  _BACKTRACE_TRIPLES "/home/calistabiondic/Documents/Prep Work/NNLS/test/CMakeLists.txt;46;ADD_TEST;/home/calistabiondic/Documents/Prep Work/NNLS/test/CMakeLists.txt;0;")
add_test(n_iter "/usr/bin/mpiexec" "-np" "1" "/home/calistabiondic/Documents/Prep Work/NNLS/build/test/Tests.exe" "nIter")
set_tests_properties(n_iter PROPERTIES  _BACKTRACE_TRIPLES "/home/calistabiondic/Documents/Prep Work/NNLS/test/CMakeLists.txt;49;ADD_TEST;/home/calistabiondic/Documents/Prep Work/NNLS/test/CMakeLists.txt;0;")
add_test(max_iter "/usr/bin/mpiexec" "-np" "1" "/home/calistabiondic/Documents/Prep Work/NNLS/build/test/Tests.exe" "maxIter")
set_tests_properties(max_iter PROPERTIES  _BACKTRACE_TRIPLES "/home/calistabiondic/Documents/Prep Work/NNLS/test/CMakeLists.txt;52;ADD_TEST;/home/calistabiondic/Documents/Prep Work/NNLS/test/CMakeLists.txt;0;")
