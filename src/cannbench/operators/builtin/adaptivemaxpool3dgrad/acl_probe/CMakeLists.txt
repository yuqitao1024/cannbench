cmake_minimum_required(VERSION 3.16)
project(adaptivemaxpool3dgrad_acl_probe)

add_executable(adaptivemaxpool3dgrad_acl_probe
    main.cpp
)

target_include_directories(adaptivemaxpool3dgrad_acl_probe PRIVATE
    $ENV{ASCEND_HOME_PATH}/include
    $ENV{ASCEND_HOME_PATH}/x86_64-linux/include
)

target_link_directories(adaptivemaxpool3dgrad_acl_probe PRIVATE
    $ENV{ASCEND_HOME_PATH}/lib64
    $ENV{ASCEND_HOME_PATH}/x86_64-linux/lib64
)

target_link_libraries(adaptivemaxpool3dgrad_acl_probe PRIVATE
    opapi
    nnopbase
    acl_rt
    ascendcl
)
