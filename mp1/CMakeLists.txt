cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(cs521_mp1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Toggle aggressive vectorization flags for o4 (disabled by default)
option(ENABLE_O4 "Enable aggressive vectorization flags for o4 (FMA/vectorize)" OFF)

if(APPLE)
    # Homebrew LLVM install prefix
    set(LLVM_HOME "/opt/homebrew/opt/llvm")

    # Force use of Homebrew LLVM compilers
    set(CMAKE_C_COMPILER "${LLVM_HOME}/bin/clang" CACHE STRING "" FORCE)
    set(CMAKE_CXX_COMPILER "${LLVM_HOME}/bin/clang++" CACHE STRING "" FORCE)

    # Add Homebrew LLVM headers and libs
    include_directories(${LLVM_HOME}/include)
    link_directories(${LLVM_HOME}/lib)

    # Tell CMake where to find OpenMP inside Homebrew LLVM
    # Note: DO NOT add -O3 here so o1–o3 use baseline optimizations
    set(OpenMP_CXX_FLAGS "-fopenmp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "${LLVM_HOME}/lib/libomp.dylib")

    message(STATUS "Using Homebrew LLVM from ${LLVM_HOME}")
else()
    # Fallback for Linux
    find_package(OpenMP REQUIRED)
    if(OpenMP_CXX_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    else()
        message(FATAL_ERROR "Need OpenMP")
    endif()
endif()

add_executable(mp1_cpu cpu/gemm_cpu.cpp)

if(APPLE)
    # Ensure OpenMP pragmas are enabled during compilation on Apple (Homebrew LLVM)
    target_compile_options(mp1_cpu PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(mp1_cpu PRIVATE ${OpenMP_omp_LIBRARY})
else()
    target_link_libraries(mp1_cpu PRIVATE OpenMP::OpenMP_CXX)
endif()

# Add aggressive flags only when ENABLE_O4 is ON
if(ENABLE_O4)
    message(STATUS "Building with ENABLE_O4: enabling -O3, -ffast-math, -march=native and related flags")
    # Common flags for both Apple/Intel and Linux
    target_compile_options(mp1_cpu PRIVATE -O3 -ffast-math -funroll-loops -ffp-contract=fast -march=native)
    if (NOT APPLE)
        # On x86_64 Linux with clang/gcc this nudges use of FMA units if available
        target_compile_options(mp1_cpu PRIVATE -mfma)
    endif()
endif()

# Convenience custom target to inspect assembly for FMA instructions
find_program(SYSTEM_OBJDUMP objdump)
if(SYSTEM_OBJDUMP)
    add_custom_target(asm
        COMMAND ${SYSTEM_OBJDUMP} -d $<TARGET_FILE:mp1_cpu> | ${CMAKE_COMMAND} -E env LC_ALL=C grep -E "fma|fmla|vfma|vfmadd|fmadd|fmla"
        DEPENDS mp1_cpu
        COMMENT "Disassembling mp1_cpu and grepping for FMA instructions"
        VERBATIM)
else()
    message(STATUS "objdump not found; skip 'asm' target")
endif()