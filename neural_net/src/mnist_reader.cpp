#include "mnist_reader.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <arpa/inet.h>

TEMPLATE mnist_reader<T>::mnist_reader() {}
TEMPLATE mnist_reader<T>::mnist_reader(unsigned int sample_size, char *image_path, char *label_path){
    init_verified(sample_size,image_path,label_path);
}

TEMPLATE void mnist_reader<T>::init(unsigned int sample_size ,char *image_path, char *label_path){
    this->sample_size = sample_size;
    FILE *image_file = fopen(image_path,"rb");
    if(!image_file){
        fprintf(stderr,"Error opening: %s\n",image_path);
        success = false;
        return;
    }
    FILE *label_file = fopen(label_path,"rb");
    if(!label_file){
        fprintf(stderr,"Error Opening: %s\n",label_path);
        success = false;
        fclose(image_file);
        return;
    }

    data = (unsigned char*)malloc(785*sample_size+784*sample_size*sizeof(T)+10*sample_size*sizeof(T));
    raw_image = data;
    raw_label = data+784*sample_size;
    processed_image = (T*)(data+785*sample_size);
    processed_label = (T*)(data+785*sample_size+784*sample_size*sizeof(T));
    memset(processed_label,0,sizeof(T)*sample_size*10);


    fseek(image_file,16,SEEK_SET);
    fseek(label_file,8,SEEK_SET);   
    
    fread(raw_image,1,784*sample_size,image_file);
    fread(raw_label,1,sample_size,label_file);

    for(int i = 0; i < sample_size; ++i) {
        for(int j = 0; j < 784; ++j) processed_image[i*784+j] = raw_image[i*784+j]/255.0;
        processed_label[i*10+raw_label[i]] = 1.0;
    }
    
    fclose(image_file); fclose(label_file);
    success = true;
}

TEMPLATE void mnist_reader<T>::init_verified(unsigned int sample_size ,char *image_path, char *label_path){
    this->sample_size = sample_size;
    FILE *image_file = fopen(image_path,"rb");
    if(!image_file){
        fprintf(stderr,"Error opening: %s\n",image_path);
        success = false;
        return;
    }
    FILE *label_file = fopen(label_path,"rb");
    if(!label_file){
        fprintf(stderr,"Error Opening: %s\n",label_path);
        success = false;
        fclose(image_file);
        return;
    }

    unsigned int *dataset_info = (unsigned int*)malloc(sizeof(int)*6);
    fread(dataset_info,sizeof(int),4,image_file);
    fread(dataset_info+4,sizeof(int),2,label_file);
    
    for(int i = 0; i < 6; ++i) dataset_info[i] = ntohl(dataset_info[i]);
    printf("Image:\nMagic num: %d\nNum Images: %d\nRows: %d\nCols: %d\nLabel:\nMagic num: %d\nNum Images: %d\n",dataset_info[0],dataset_info[1],dataset_info[2],dataset_info[3],dataset_info[4],dataset_info[5]);
    if(dataset_info[0] == 2051 && (dataset_info[1] == 60000 || dataset_info[1] == 10000) && dataset_info[2] == 28 && dataset_info[3] == 28 && dataset_info[4] == 2049 && (dataset_info[5] == 60000 || dataset_info[5] == 10000)){
        printf("Verified\n");
    }else{
        fprintf(stderr,"Dataset is not correct\n");
        return;
    }

    if(sample_size > dataset_info[1]){
        fprintf(stderr,"Sample size bigger then dataset sample size\n");
        return;
    }   
    
    data = (unsigned char*)malloc(785*sample_size+784*sample_size*sizeof(T)+10*sample_size*sizeof(T));
    raw_image = data;
    raw_label = data+784*sample_size;
    processed_image = (T*)(data+785*sample_size);
    processed_label = (T*)(data+785*sample_size+784*sample_size*sizeof(T));
    memset(processed_label,0,sizeof(T)*sample_size*10);

    free(dataset_info);

    fread(raw_image,1,784*sample_size,image_file);
    fread(raw_label,1,sample_size,label_file);

    for(int i = 0; i < sample_size; ++i) {
        for(int j = 0; j < 784; ++j) processed_image[i*784+j] = raw_image[i*784+j]/255.0;
        processed_label[i*10+raw_label[i]] = 1.0;
    }
    
    fclose(image_file); fclose(label_file);
    success = true;
}

TEMPLATE unsigned char * mnist_reader<T>::get_label(){
    return raw_label;
}

TEMPLATE unsigned char * mnist_reader<T>::get_image(){
    return raw_image;
}

TEMPLATE T *mnist_reader<T>::get_processed_image(){
    return processed_image;
}

TEMPLATE T *mnist_reader<T>::get_processed_label(){
    return processed_label;
}

TEMPLATE mnist_reader<T>::~mnist_reader(){
    if(success) free(data);
}

template class mnist_reader<double>;
template class mnist_reader<float>;