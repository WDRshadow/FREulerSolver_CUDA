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

add_executable(jacobian_test
        jacobian_test.cpp
)
target_link_libraries(jacobian_test
        PRIVATE
        GTest::gtest_main
        fr
)
add_test(NAME jacobian_test COMMAND jacobian_test)

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

# -----------------------------------------

add_executable(solver_test
        solver_test.cpp
)
target_link_libraries(solver_test
        PRIVATE
        GTest::gtest_main
        fr
)
add_test(NAME solver_test COMMAND solver_test)

# -----------------------------------------

add_executable(visualize_test
        visualize_test.cpp
)
target_link_libraries(visualize_test
        PRIVATE
        GTest::gtest_main
        fr
)
add_test(NAME visualize_test COMMAND visualize_test)

# -----------------------------------------

add_executable(limiter_test
        limiter_test.cpp
)
target_link_libraries(limiter_test
        PRIVATE
        GTest::gtest_main
        fr
)
add_test(NAME limiter_test COMMAND limiter_test)
