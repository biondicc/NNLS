# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Produce verbose output by default.
VERBOSE = 1

# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/home/calistabiondic/Documents/Prep Work/NNLS"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/home/calistabiondic/Documents/Prep Work/NNLS/build"

# Include any dependencies generated for this target.
include test/CMakeFiles/Tests.exe.dir/depend.make

# Include the progress variables for this target.
include test/CMakeFiles/Tests.exe.dir/progress.make

# Include the compile flags for this target's objects.
include test/CMakeFiles/Tests.exe.dir/flags.make

test/CMakeFiles/Tests.exe.dir/test_cases.cpp.o: test/CMakeFiles/Tests.exe.dir/flags.make
test/CMakeFiles/Tests.exe.dir/test_cases.cpp.o: ../test/test_cases.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/calistabiondic/Documents/Prep Work/NNLS/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object test/CMakeFiles/Tests.exe.dir/test_cases.cpp.o"
	cd "/home/calistabiondic/Documents/Prep Work/NNLS/build/test" && /usr/bin/mpicxx  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Tests.exe.dir/test_cases.cpp.o -c "/home/calistabiondic/Documents/Prep Work/NNLS/test/test_cases.cpp"

test/CMakeFiles/Tests.exe.dir/test_cases.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Tests.exe.dir/test_cases.cpp.i"
	cd "/home/calistabiondic/Documents/Prep Work/NNLS/build/test" && /usr/bin/mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/calistabiondic/Documents/Prep Work/NNLS/test/test_cases.cpp" > CMakeFiles/Tests.exe.dir/test_cases.cpp.i

test/CMakeFiles/Tests.exe.dir/test_cases.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Tests.exe.dir/test_cases.cpp.s"
	cd "/home/calistabiondic/Documents/Prep Work/NNLS/build/test" && /usr/bin/mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/calistabiondic/Documents/Prep Work/NNLS/test/test_cases.cpp" -o CMakeFiles/Tests.exe.dir/test_cases.cpp.s

test/CMakeFiles/Tests.exe.dir/__/src/NNLS_solver.cpp.o: test/CMakeFiles/Tests.exe.dir/flags.make
test/CMakeFiles/Tests.exe.dir/__/src/NNLS_solver.cpp.o: ../src/NNLS_solver.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/calistabiondic/Documents/Prep Work/NNLS/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object test/CMakeFiles/Tests.exe.dir/__/src/NNLS_solver.cpp.o"
	cd "/home/calistabiondic/Documents/Prep Work/NNLS/build/test" && /usr/bin/mpicxx  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Tests.exe.dir/__/src/NNLS_solver.cpp.o -c "/home/calistabiondic/Documents/Prep Work/NNLS/src/NNLS_solver.cpp"

test/CMakeFiles/Tests.exe.dir/__/src/NNLS_solver.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Tests.exe.dir/__/src/NNLS_solver.cpp.i"
	cd "/home/calistabiondic/Documents/Prep Work/NNLS/build/test" && /usr/bin/mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/calistabiondic/Documents/Prep Work/NNLS/src/NNLS_solver.cpp" > CMakeFiles/Tests.exe.dir/__/src/NNLS_solver.cpp.i

test/CMakeFiles/Tests.exe.dir/__/src/NNLS_solver.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Tests.exe.dir/__/src/NNLS_solver.cpp.s"
	cd "/home/calistabiondic/Documents/Prep Work/NNLS/build/test" && /usr/bin/mpicxx $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/calistabiondic/Documents/Prep Work/NNLS/src/NNLS_solver.cpp" -o CMakeFiles/Tests.exe.dir/__/src/NNLS_solver.cpp.s

# Object files for target Tests.exe
Tests_exe_OBJECTS = \
"CMakeFiles/Tests.exe.dir/test_cases.cpp.o" \
"CMakeFiles/Tests.exe.dir/__/src/NNLS_solver.cpp.o"

# External object files for target Tests.exe
Tests_exe_EXTERNAL_OBJECTS =

test/Tests.exe: test/CMakeFiles/Tests.exe.dir/test_cases.cpp.o
test/Tests.exe: test/CMakeFiles/Tests.exe.dir/__/src/NNLS_solver.cpp.o
test/Tests.exe: test/CMakeFiles/Tests.exe.dir/build.make
test/Tests.exe: /home/calistabiondic/PHiLiP_ETC/Libraries/Trilinos/install/lib/librol.so.13.1
test/Tests.exe: /home/calistabiondic/PHiLiP_ETC/Libraries/Trilinos/install/lib/libintrepid.so.13.1
test/Tests.exe: /home/calistabiondic/PHiLiP_ETC/Libraries/Trilinos/install/lib/libbelosepetra.so.13.1
test/Tests.exe: /home/calistabiondic/PHiLiP_ETC/Libraries/Trilinos/install/lib/libbelos.so.13.1
test/Tests.exe: /home/calistabiondic/PHiLiP_ETC/Libraries/Trilinos/install/lib/libml.so.13.1
test/Tests.exe: /home/calistabiondic/PHiLiP_ETC/Libraries/Trilinos/install/lib/libifpack.so.13.1
test/Tests.exe: /home/calistabiondic/PHiLiP_ETC/Libraries/Trilinos/install/lib/libpamgen_extras.so.13.1
test/Tests.exe: /home/calistabiondic/PHiLiP_ETC/Libraries/Trilinos/install/lib/libpamgen.so.13.1
test/Tests.exe: /home/calistabiondic/PHiLiP_ETC/Libraries/Trilinos/install/lib/libamesos.so.13.1
test/Tests.exe: /home/calistabiondic/PHiLiP_ETC/Libraries/Trilinos/install/lib/libgaleri-epetra.so.13.1
test/Tests.exe: /home/calistabiondic/PHiLiP_ETC/Libraries/Trilinos/install/lib/libaztecoo.so.13.1
test/Tests.exe: /home/calistabiondic/PHiLiP_ETC/Libraries/Trilinos/install/lib/libisorropia.so.13.1
test/Tests.exe: /home/calistabiondic/PHiLiP_ETC/Libraries/Trilinos/install/lib/libtrilinosss.so.13.1
test/Tests.exe: /home/calistabiondic/PHiLiP_ETC/Libraries/Trilinos/install/lib/libepetraext.so.13.1
test/Tests.exe: /home/calistabiondic/PHiLiP_ETC/Libraries/Trilinos/install/lib/libtriutils.so.13.1
test/Tests.exe: /home/calistabiondic/PHiLiP_ETC/Libraries/Trilinos/install/lib/libshards.so.13.1
test/Tests.exe: /home/calistabiondic/PHiLiP_ETC/Libraries/Trilinos/install/lib/libzoltan.so.13.1
test/Tests.exe: /home/calistabiondic/PHiLiP_ETC/Libraries/Trilinos/install/lib/libepetra.so.13.1
test/Tests.exe: /home/calistabiondic/PHiLiP_ETC/Libraries/Trilinos/install/lib/libsacado.so.13.1
test/Tests.exe: /home/calistabiondic/PHiLiP_ETC/Libraries/Trilinos/install/lib/libteuchosremainder.so.13.1
test/Tests.exe: /home/calistabiondic/PHiLiP_ETC/Libraries/Trilinos/install/lib/libteuchosnumerics.so.13.1
test/Tests.exe: /home/calistabiondic/PHiLiP_ETC/Libraries/Trilinos/install/lib/libteuchoscomm.so.13.1
test/Tests.exe: /home/calistabiondic/PHiLiP_ETC/Libraries/Trilinos/install/lib/libteuchosparameterlist.so.13.1
test/Tests.exe: /home/calistabiondic/PHiLiP_ETC/Libraries/Trilinos/install/lib/libteuchosparser.so.13.1
test/Tests.exe: /home/calistabiondic/PHiLiP_ETC/Libraries/Trilinos/install/lib/libteuchoscore.so.13.1
test/Tests.exe: /home/calistabiondic/PHiLiP_ETC/Libraries/Trilinos/install/lib/libgtest.so.13.1
test/Tests.exe: /usr/lib/x86_64-linux-gnu/libopenblas.so
test/Tests.exe: /usr/lib/x86_64-linux-gnu/libopenblas.so
test/Tests.exe: /usr/lib/x86_64-linux-gnu/libopenblas.so
test/Tests.exe: /usr/lib/x86_64-linux-gnu/libopenblas.so
test/Tests.exe: /usr/lib/x86_64-linux-gnu/libopenblas.so
test/Tests.exe: /usr/lib/x86_64-linux-gnu/libopenblas.so
test/Tests.exe: /usr/lib/x86_64-linux-gnu/libopenblas.so
test/Tests.exe: /usr/lib/x86_64-linux-gnu/libopenblas.so
test/Tests.exe: /usr/lib/x86_64-linux-gnu/libopenblas.so
test/Tests.exe: /usr/lib/x86_64-linux-gnu/libopenblas.so
test/Tests.exe: /usr/lib/x86_64-linux-gnu/libopenblas.so
test/Tests.exe: /usr/lib/x86_64-linux-gnu/libopenblas.so
test/Tests.exe: /usr/lib/x86_64-linux-gnu/libopenblas.so
test/Tests.exe: /usr/lib/x86_64-linux-gnu/libopenblas.so
test/Tests.exe: /usr/lib/x86_64-linux-gnu/libopenblas.so
test/Tests.exe: /usr/lib/x86_64-linux-gnu/libopenblas.so
test/Tests.exe: /usr/lib/x86_64-linux-gnu/libopenblas.so
test/Tests.exe: /usr/lib/x86_64-linux-gnu/libopenblas.so
test/Tests.exe: /usr/lib/x86_64-linux-gnu/libopenblas.so
test/Tests.exe: /usr/lib/x86_64-linux-gnu/libopenblas.so
test/Tests.exe: /usr/lib/x86_64-linux-gnu/libopenblas.so
test/Tests.exe: /usr/lib/x86_64-linux-gnu/libopenblas.so
test/Tests.exe: /usr/lib/x86_64-linux-gnu/libopenblas.so
test/Tests.exe: /usr/lib/x86_64-linux-gnu/libopenblas.so
test/Tests.exe: /usr/lib/x86_64-linux-gnu/libopenblas.so
test/Tests.exe: /usr/lib/x86_64-linux-gnu/libopenblas.so
test/Tests.exe: /usr/lib/x86_64-linux-gnu/libopenblas.so
test/Tests.exe: /usr/lib/x86_64-linux-gnu/libopenblas.so
test/Tests.exe: /usr/lib/x86_64-linux-gnu/libopenblas.so
test/Tests.exe: /usr/lib/x86_64-linux-gnu/libopenblas.so
test/Tests.exe: test/CMakeFiles/Tests.exe.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/home/calistabiondic/Documents/Prep Work/NNLS/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable Tests.exe"
	cd "/home/calistabiondic/Documents/Prep Work/NNLS/build/test" && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Tests.exe.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/CMakeFiles/Tests.exe.dir/build: test/Tests.exe

.PHONY : test/CMakeFiles/Tests.exe.dir/build

test/CMakeFiles/Tests.exe.dir/clean:
	cd "/home/calistabiondic/Documents/Prep Work/NNLS/build/test" && $(CMAKE_COMMAND) -P CMakeFiles/Tests.exe.dir/cmake_clean.cmake
.PHONY : test/CMakeFiles/Tests.exe.dir/clean

test/CMakeFiles/Tests.exe.dir/depend:
	cd "/home/calistabiondic/Documents/Prep Work/NNLS/build" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/home/calistabiondic/Documents/Prep Work/NNLS" "/home/calistabiondic/Documents/Prep Work/NNLS/test" "/home/calistabiondic/Documents/Prep Work/NNLS/build" "/home/calistabiondic/Documents/Prep Work/NNLS/build/test" "/home/calistabiondic/Documents/Prep Work/NNLS/build/test/CMakeFiles/Tests.exe.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : test/CMakeFiles/Tests.exe.dir/depend

