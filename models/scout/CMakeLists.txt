# Find Pybind11 supports
find_package(pybind11 REQUIRED)

pybind11_add_module(
    flame_scout
    MODULE flame_scout.cpp
)

target_link_libraries(
    flame_scout
    PRIVATE dv::processing
            dv::toolkit
)
