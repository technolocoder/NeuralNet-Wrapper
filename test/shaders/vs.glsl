#version 420 core

layout(location=0) in vec2 position;
layout(location=1) in vec2 offset;
layout(location=2) in float color;

out VS_OUT{
    float color;
} vs_out;

void main(){
    gl_Position = vec4(vec2(position+offset),0.0,1.0);
    vs_out.color = color;
}