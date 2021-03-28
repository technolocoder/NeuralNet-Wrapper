add_executable(image_load image_load.cpp)
target_link_libraries(image_load PUBLIC neural_net)

add_executable(image_render image_render.cpp)
target_link_libraries(image_render PUBLIC neural_net GL GLU GLEW SDL2)
target_link_directories(image_render PUBLIC /usr/local/lib)

add_executable(image_pool image_pool.cpp)
target_link_libraries(image_pool PUBLIC neural_net GL GLU GLEW SDL2)
target_link_directories(image_pool PUBLIC /usr/local/lib)

add_executable(image_conv image_conv.cpp)
target_link_libraries(image_conv PUBLIC neural_net GL GLU GLEW SDL2)
target_link_directories(image_conv PUBLIC /usr/local/lib)

add_executable(image_feedforward image_feedforward.cpp)
target_link_libraries(image_feedforward PUBLIC neural_net GL GLU GLEW SDL2)
target_link_directories(image_feedforward PUBLIC /usr/local/lib)

add_executable(net_memcheck net_memcheck.cpp)
target_link_libraries(net_memcheck PUBLIC neural_net GL GLU GLEW SDL2)
target_link_directories(net_memcheck PUBLIC /usr/local/lib)