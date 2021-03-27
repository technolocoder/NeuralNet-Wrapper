#version 420 core

in VS_OUT{
    float color;
} fs_in;

out vec4 color;

void main(){
    color = vec4(fs_in.color);
}