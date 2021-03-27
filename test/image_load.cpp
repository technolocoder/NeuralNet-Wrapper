#include "mnist_reader.hpp"
using namespace std;

int main(){
    int sample_size = 60000;
    char train_images[] = "test/dataset/train-images", train_labels[] = "test/dataset/train-labels";
    mnist_reader<float> reader(sample_size,train_images,train_labels);
    return 0;
}