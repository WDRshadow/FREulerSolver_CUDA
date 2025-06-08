find_package(GTest REQUIRED)

# -----------------------------------------

add_executable(mesh_init_test
        mesh_init_test.cpp
)
target_link_libraries(mesh_init_test
        PRIVATE
        GTest::gtest_main
        fr
)
add_test(NAME mesh_init_test COMMAND mesh_init_test)

# -----------------------------------------

add_executable(nodes_init_test
        nodes_init_test.cpp
)
target_link_libraries(nodes_init_test
        PRIVATE
        GTest::gtest_main
        fr
)
add_test(NAME nodes_init_test COMMAND nodes_init_test)

# -----------------------------------------

add_executable(flux_utils_test
        flux_utils_test.cpp
)
target_link_libraries(flux_utils_test
        PRIVATE
        GTest::gtest_main
        fr
)
add_test(NAME flux_utils_test COMMAND flux_utils_test)