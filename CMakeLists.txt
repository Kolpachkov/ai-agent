cmake_minimum_required(VERSION 3.16)
project(ai-agent VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ── System dependencies ───────────────────────────────────────────────────────
find_package(llama REQUIRED)
find_package(nlohmann_json REQUIRED)

# ── Executable ────────────────────────────────────────────────────────────────
add_executable(ai-agent
    src/main.cpp
    src/agent.cpp
    src/config.cpp
    src/tools.cpp
)

target_include_directories(ai-agent PRIVATE src/)

find_library(READLINE_LIB readline REQUIRED)

target_link_libraries(ai-agent PRIVATE
    llama
    nlohmann_json::nlohmann_json
    ${READLINE_LIB}
)

target_compile_options(ai-agent PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -march=native -Wall -Wextra>
    $<$<CXX_COMPILER_ID:MSVC>:/O2 /W3>
)

# ── Tests (no llama dependency) ───────────────────────────────────────────────
enable_testing()

# Shared tools object (no llama, no readline)
add_library(tools-lib OBJECT src/tools.cpp)
target_include_directories(tools-lib PUBLIC src/)
target_link_libraries(tools-lib PUBLIC nlohmann_json::nlohmann_json)
target_compile_options(tools-lib PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-O2 -Wall -Wextra>)

add_executable(test_parsing tests/test_parsing.cpp)
target_include_directories(test_parsing PRIVATE src/)
target_compile_options(test_parsing PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-O0 -g -Wall -Wextra>)
add_test(NAME test_parsing COMMAND test_parsing)

add_executable(test_tools tests/test_tools.cpp $<TARGET_OBJECTS:tools-lib>)
target_include_directories(test_tools PRIVATE src/)
target_link_libraries(test_tools PRIVATE nlohmann_json::nlohmann_json)
target_compile_options(test_tools PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-O0 -g -Wall -Wextra>)
add_test(NAME test_tools COMMAND test_tools)
