#ifndef __MNIST_READER_WRAPPER_
#define __MNIST_READER_WRAPPER_

#include "common.hpp"

TEMPLATE class mnist_reader{
public:
    mnist_reader();
    mnist_reader(unsigned int sample_size, char *image_path, char *label_path);

    void init(unsigned int sample_size, char *image_path, char *label_path);
    void init_verified(unsigned int sample_size ,char *image_path ,char *label_path);

    unsigned char *get_label();
    unsigned char *get_image();

    T *get_processed_label();
    T *get_processed_image();

    ~mnist_reader();
private:
    int sample_size,offsets=0,total_size;
    bool success = false;
    unsigned char *raw_image, *raw_label, *data;
    T *processed_image, *processed_label; 
};

#endif 