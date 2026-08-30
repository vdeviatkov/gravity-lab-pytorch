cmake_minimum_required(VERSION 3.24)
project(GravityLabAIArcade VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Keep outputs where the Python binding and convenience scripts already discover them.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/gravity-lab/build-classic-rl")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/gravity-lab/build-classic-rl")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/gravity-lab/build-classic-rl")

set(GRAVITY_LAB_BUILD_CLASSIC ON CACHE BOOL "" FORCE)
set(GRAVITY_LAB_BUILD_DESKTOP OFF CACHE BOOL "" FORCE)
set(GRAVITY_LAB_BUILD_TESTS ON CACHE BOOL "" FORCE)
add_subdirectory(gravity-lab gravity-lab-build)

find_package(PkgConfig REQUIRED)
pkg_search_module(AI_SDL2 REQUIRED IMPORTED_TARGET sdl2)
pkg_search_module(AI_SDL2_TTF REQUIRED IMPORTED_TARGET SDL2_ttf)

add_executable(gravity_lab_ai_arcade apps/ai_arcade.cpp)
target_link_libraries(gravity_lab_ai_arcade PRIVATE
    gravity_lab_classic_renderer
    gravity_lab_policy
    PkgConfig::AI_SDL2
    PkgConfig::AI_SDL2_TTF
)
target_compile_definitions(gravity_lab_ai_arcade PRIVATE
    GRAVITY_LAB_AI_FONT_PATH="${CMAKE_SOURCE_DIR}/gravity-lab/classic/assets/FontSansSerif.ttf"
)

