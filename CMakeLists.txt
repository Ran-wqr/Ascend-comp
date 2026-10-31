cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)

project(nvcomp_ascend LANGUAGES ASC CXX)

set(NPU_ARCH "dav-2201" CACHE STRING "npu-arch")

# ---- demo / CLI tools -------------------------------------------------------
add_executable(lz4_demo    src/lz4_ascend.asc)
add_executable(snappy_demo src/snappy_ascend.asc)

# ---- correctness tests (mirrors nvCOMP tests/) ------------------------------
add_executable(test_lz4    tests/test_lz4.asc)
add_executable(test_snappy tests/test_snappy.asc)

# ---- benchmarks (mirrors nvCOMP benchmarks/) --------------------------------
add_executable(bench_lz4    benchmarks/bench_lz4.asc)
add_executable(bench_snappy benchmarks/bench_snappy.asc)
add_executable(bench_fp     benchmarks/bench_fp.asc)
add_executable(bench_lz4_file benchmarks/bench_lz4_file.asc)

set(ALL_TARGETS lz4_demo snappy_demo test_lz4 test_snappy bench_lz4 bench_snappy bench_fp bench_lz4_file)

foreach(t ${ALL_TARGETS})
    target_include_directories(${t} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/tests
        ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks)
    target_compile_options(${t} PRIVATE
        $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${NPU_ARCH}>)
    target_link_libraries(${t} PRIVATE m)
endforeach()

# bench_fp also drives the official HANS op through aclnn (host API)
target_include_directories(bench_fp PRIVATE
    /usr/local/Ascend/cann-9.0.0/aarch64-linux/include)
target_link_directories(bench_fp PRIVATE
    /usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64
    /usr/local/Ascend/cann-9.0.0/lib64)
target_link_libraries(bench_fp PRIVATE opapi_math nnopbase ascendcl)
