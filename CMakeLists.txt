cmake_minimum_required (VERSION 3.31)

include($ENV{ARIEO_BUILDENV_PACKAGE_INSTALL_FOLDER}/cmake/build_environment.cmake)
project(ArieoEngine)

arieo_engine_project(
    arieo_render_test_module
    PROJECT_TYPE module

    PACKAGES
        Arieo-Interface-Sample
        Arieo-Core
    INTERFACES
        Arieo-Interface-Sample::arieo_sample_interface
        Arieo-Interface-RHI::arieo_interface_rhi
        Arieo-Interface-Archive::arieo_interface_archive
        Arieo-Interface-FileLoader::arieo_interface_file_loader
        Arieo-Interface-Main::arieo_interface_main
    PRIVATE_LIBS
        Arieo-Core::arieo_core

    SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/private/src/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/private/src/*/*.cpp
)