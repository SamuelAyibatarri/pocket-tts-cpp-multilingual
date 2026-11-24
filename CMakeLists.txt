# pocket-tts-cpp-multilingual — Cross-platform C++ FFI runtime for Kyutai Pocket TTS
# Based on VolgaGerm/PocketTTS.cpp (MIT License)

cmake_minimum_required(VERSION 3.24)
project(pocket_tts VERSION 1.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

# ONNX Runtime (pre-built) — https://github.com/microsoft/onnxruntime/releases
# Pass -DLOCAL_ORT_ROOT=/path/to/ort to use a pre-staged ONNX Runtime instead of downloading.
set(ONNXRUNTIME_VERSION "1.21.0" CACHE STRING "ONNX Runtime release version")
set(LOCAL_ORT_ROOT "" CACHE PATH "Optional pre-staged ONNX Runtime root (include/ + lib/)")

if(LOCAL_ORT_ROOT)
    set(ONNXRUNTIME_ROOT ${LOCAL_ORT_ROOT})
else()
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
            set(ONNXRUNTIME_ARCH "x64")
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
            set(ONNXRUNTIME_ARCH "aarch64")
        endif()
        set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-${ONNXRUNTIME_ARCH}-${ONNXRUNTIME_VERSION}.tgz")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-osx-universal2-${ONNXRUNTIME_VERSION}.tgz")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-win-x64-${ONNXRUNTIME_VERSION}.zip")
    endif()

    FetchContent_Declare(onnxruntime
        URL ${ONNXRUNTIME_URL}
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(onnxruntime)
    set(ONNXRUNTIME_ROOT ${onnxruntime_SOURCE_DIR})
endif()

set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_ROOT}/include)
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(ONNXRUNTIME_LIB ${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib)
    set(ONNXRUNTIME_RUNTIME_LIB ${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(ONNXRUNTIME_LIB ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.dylib)
    set(ONNXRUNTIME_RUNTIME_LIB ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.dylib)
else()
    set(ONNXRUNTIME_LIB ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so)
    set(ONNXRUNTIME_RUNTIME_LIB ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so)
endif()

# SentencePiece (static library)
set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)
set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL "" FORCE)
set(SPM_PROTOBUF_PROVIDER "internal" CACHE STRING "" FORCE)
set(SPM_ABSL_PROVIDER "internal" CACHE STRING "" FORCE)
set(SPM_ENABLE_NFKC_COMPILE OFF CACHE BOOL "" FORCE)

file(WRITE ${CMAKE_BINARY_DIR}/patch_sentencepiece.cmake "
file(READ \"\${SOURCE_DIR}/src/sentencepiece_processor.h\" CONTENT)
if(NOT CONTENT MATCHES \"#include <cstdint>\")
    string(REPLACE \"#include <string>\" \"#include <string>\\n#include <cstdint>\" CONTENT \"\${CONTENT}\")
    file(WRITE \"\${SOURCE_DIR}/src/sentencepiece_processor.h\" \"\${CONTENT}\")
endif()
file(READ \"\${SOURCE_DIR}/CMakeLists.txt\" CMLCONTENT)
string(REGEX REPLACE \"cmake_minimum_required\\\\(VERSION [0-9.]+\\\\)\" \"cmake_minimum_required(VERSION 3.14)\" CMLCONTENT \"\${CMLCONTENT}\")
file(WRITE \"\${SOURCE_DIR}/CMakeLists.txt\" \"\${CMLCONTENT}\")
")

FetchContent_Declare(sentencepiece
    GIT_REPOSITORY https://github.com/google/sentencepiece.git
    GIT_TAG v0.2.1
    GIT_SHALLOW TRUE
    PATCH_COMMAND ${CMAKE_COMMAND} -DSOURCE_DIR=<SOURCE_DIR> -P ${CMAKE_BINARY_DIR}/patch_sentencepiece.cmake
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(sentencepiece)
if(NOT MSVC)
    target_compile_options(sentencepiece-static PRIVATE -w)
endif()
set(SENTENCEPIECE_INCLUDE_DIRS "${sentencepiece_SOURCE_DIR}/src")

# dr_libs (header-only)
FetchContent_Declare(dr_libs
    GIT_REPOSITORY https://github.com/mackron/dr_libs.git
    GIT_TAG master
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(dr_libs)

option(BUILD_SHARED_LIB "Build libpocket_tts shared library for FFI (Python, Deno, etc.)" ON)

# CLI / HTTP server executable
add_executable(pocket-tts pocket_tts.cpp)
target_include_directories(pocket-tts PRIVATE
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${SENTENCEPIECE_INCLUDE_DIRS}
    ${dr_libs_SOURCE_DIR}
)
target_link_libraries(pocket-tts PRIVATE ${ONNXRUNTIME_LIB} sentencepiece-static)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(pocket-tts PRIVATE pthread dl)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    target_link_libraries(pocket-tts PRIVATE ws2_32)
endif()
add_custom_command(TARGET pocket-tts POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ONNXRUNTIME_RUNTIME_LIB} $<TARGET_FILE_DIR:pocket-tts>
)

# Shared library for FFI
if(BUILD_SHARED_LIB)
    add_library(pocket-tts-lib SHARED pocket_tts.cpp)
    set_target_properties(pocket-tts-lib PROPERTIES
        OUTPUT_NAME "pocket_tts"
        WINDOWS_EXPORT_ALL_SYMBOLS TRUE
    )
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        set_target_properties(pocket-tts-lib PROPERTIES
            BUILD_RPATH "$ORIGIN"
            INSTALL_RPATH "$ORIGIN"
            BUILD_WITH_INSTALL_RPATH TRUE
        )
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        set_target_properties(pocket-tts-lib PROPERTIES
            BUILD_RPATH "@loader_path"
            INSTALL_RPATH "@loader_path"
            BUILD_WITH_INSTALL_RPATH TRUE
        )
    endif()

    target_compile_definitions(pocket-tts-lib PRIVATE PTT_SHARED_LIB)
    target_include_directories(pocket-tts-lib PRIVATE
        ${ONNXRUNTIME_INCLUDE_DIR}
        ${SENTENCEPIECE_INCLUDE_DIRS}
        ${dr_libs_SOURCE_DIR}
    )
    target_link_libraries(pocket-tts-lib PRIVATE ${ONNXRUNTIME_LIB} sentencepiece-static)

    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        target_link_libraries(pocket-tts-lib PRIVATE pthread dl)
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        target_link_libraries(pocket-tts-lib PRIVATE ws2_32)
    endif()

    add_custom_command(TARGET pocket-tts-lib POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ONNXRUNTIME_RUNTIME_LIB} $<TARGET_FILE_DIR:pocket-tts-lib>
    )
endif()
