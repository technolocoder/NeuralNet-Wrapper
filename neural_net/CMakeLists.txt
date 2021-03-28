set(FILES "include/layer/IL.hpp;include/layer/PL.hpp;include/layer/layer.hpp;src/layer/IL.cpp;src/layer/PL.cpp;src/mnist_reader;include/mnist_reader.hpp;include/neural_network.hpp;src/neural_network.cpp;src/layer/CL.cpp;include/layer/CL.hpp;include/layer/FL.hpp;src/layer/FL.cpp;include/layer/FC.hpp;src/layer/FC.cpp;src/layer/SL.cpp;include/layer/SL.hpp")

add_library(neural_net SHARED "${FILES}")
target_include_directories(neural_net PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include;${CMAKE_CURRENT_SOURCE_DIR}/include/layer")
target_precompile_headers(neural_net PUBLIC include/activations.hpp)