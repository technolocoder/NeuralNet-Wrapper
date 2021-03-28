#include "neural_network.hpp"
#include "mnist_reader.hpp"

int main(){
    int sample_size = 100;
    mnist_reader<float> reader(sample_size,"test/dataset/train-images","test/dataset/train-labels");
    neural_network<float> net(sample_size,9);
    net.add_input_layer(28,28,1,false);
    net.add_convolution_layer(3,3,1,1,10,TANH);
    net.add_pooling_layer(2,2,2,2,MAX_POOL);
    net.add_convolution_layer(5,5,1,1,20,TANH);
    net.add_pooling_layer(2,2,2,2,MAX_POOL);
    net.add_flatten_layer();
    net.add_fully_connected(20,TANH);
    net.add_fully_connected(10,TANH);
    net.add_softmax_layer();
    net.construct_neuralnet();

    float *output = net.feedforward(reader.get_processed_image());
    return 0;
}