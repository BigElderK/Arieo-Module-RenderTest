cmake_minimum_required (VERSION 3.31)

include($env{ARIEO_BUILD_ENV_PACKAGES_OUTPUT_DIR}/cmake/build_environment.cmake)
project(ArieoEngine)

arieo_engine_project(
    arieo_render_test_module
    PROJECT_TYPE module

    DEPENDENCIES
        ARIEO_PACKAGES
            Arieo-Core
            Arieo-Interface-RHI
            Arieo-Interface-Window
            Arieo-Interface-Archive
            Arieo-Interface-FileLoader
            Arieo-Interface-Main
        INTERFACES
            Arieo-Interface-RHI::arieo_rhi_interface
            Arieo-Interface-Archive::arieo_archive_interface
            Arieo-Interface-FileLoader::arieo_file_loader_interface
            Arieo-Interface-Main::arieo_main_interface
        PRIVATE_LIBS
            Arieo-Core::arieo_core

    SOURCES
        CXX_SOURCES
            ${CMAKE_CURRENT_LIST_DIR}/private/src/*.cpp
            ${CMAKE_CURRENT_LIST_DIR}/private/src/*/*.cpp
)
