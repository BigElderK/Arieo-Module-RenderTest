cmake_minimum_required (VERSION 3.31)

include($ENV{ARIEO_BUILD_ENV_PACKAGES_OUTPUT_DIR}/cmake/build_environment.cmake)

ARIEO_PACKAGE(
    Arieo-Module-RenderTest
    CATEGORY 02_engine
    DEPENDS
        https://github.com/BigElderK/Arieo-Core.git@main
        https://github.com/BigElderK/Arieo-Interface-RHI.git@main
        https://github.com/BigElderK/Arieo-Interface-Window.git@main
        https://github.com/BigElderK/Arieo-Interface-Archive.git@main
        https://github.com/BigElderK/Arieo-Interface-FileLoader.git@main
        https://github.com/BigElderK/Arieo-Interface-Main.git@main
)

ARIEO_ENGINE_PROJECT(
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
        PRIVATE_LIBS
            Arieo-Interface-RHI::arieo_rhi_interface
            Arieo-Interface-Archive::arieo_archive_interface
            Arieo-Interface-FileLoader::arieo_file_loader_interface
            Arieo-Interface-Main::arieo_main_interface
            Arieo-Core::arieo_core

    SOURCES
        CXX_SOURCES
            ${CMAKE_CURRENT_LIST_DIR}/private/src/*.cpp
            ${CMAKE_CURRENT_LIST_DIR}/private/src/*/*.cpp
)
