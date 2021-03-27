#include "neural_network.hpp"
#include "mnist_reader.hpp"
#include <SDL2/SDL.h>
#include <GL/glew.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>

using namespace std;
GLuint create_shader(const char *shader_path, GLenum shader_type){
    ifstream filestream(shader_path,ios::in);
    if(!filestream.is_open()){
        cerr << "Error opening file named: " << shader_path << '\n';
        return -1;
    }
    
    ostringstream ss;
    ss << filestream.rdbuf();
    string str = ss.str();
    const char *src = str.c_str();
    
    GLuint shader = glCreateShader(shader_type);
    glShaderSource(shader,1,&src,NULL);
    glCompileShader(shader);
    
    int compile_status;
    glGetShaderiv(shader,GL_COMPILE_STATUS,&compile_status);

    if(!compile_status){
        int log_length;
        glGetShaderiv(shader,GL_INFO_LOG_LENGTH,&log_length);
        char log[log_length];
        glGetShaderInfoLog(shader,log_length,NULL,log);
        cerr << log << '\n';
        return -1;
    }
    filestream.close();

    return shader;
}   

GLuint create_program(const char *vs_shader_path, const char *fs_shader_path){
    GLuint program_id = glCreateProgram();
    glAttachShader(program_id,create_shader(vs_shader_path,GL_VERTEX_SHADER));
    glAttachShader(program_id,create_shader(fs_shader_path,GL_FRAGMENT_SHADER));
    glLinkProgram(program_id);

    int link_status;
    glGetProgramiv(program_id,GL_LINK_STATUS,&link_status);

    if(!link_status){
        int log_length;
        glGetProgramiv(program_id,GL_INFO_LOG_LENGTH,&log_length);
        char log[log_length];
        glGetProgramInfoLog(program_id,log_length,NULL,log);
        cerr << log << '\n';
        return -1;
    }
    return program_id;
}

int main(){
    SDL_Init(SDL_INIT_VIDEO);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION,4);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION,2);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK,SDL_GL_CONTEXT_PROFILE_CORE);

    SDL_DisplayMode display_mode;
    SDL_GetDesktopDisplayMode(0,&display_mode);

    const int window_width = display_mode.w , window_height = display_mode.h ,frame_delay = 1000000/display_mode.refresh_rate;
    
    SDL_Window *window = SDL_CreateWindow("MNIST Render",SDL_WINDOWPOS_UNDEFINED,SDL_WINDOWPOS_UNDEFINED,window_width,window_height,SDL_WINDOW_OPENGL|SDL_WINDOW_FULLSCREEN_DESKTOP);
    SDL_GLContext context = SDL_GL_CreateContext(window);

    glewExperimental = GL_TRUE;
    glewInit();

    GLuint program = create_program("test/shaders/vs.glsl","test/shaders/fs.glsl");
    const int rows = 28, cols = 28 ,sample_size = 60000;
    
    mnist_reader<float> reader(sample_size,"test/dataset/train-images","test/dataset/train-labels");

    neural_network<float> net(sample_size,2);
    net.add_input_layer(cols,rows,1,false);
    net.add_pooling_layer(2,2,2,2,MAX_POOL);
    net.construct_neuralnet();

    dim output_dim = net.get_output_dimensions();
    const int o_rows = output_dim.x, o_cols = output_dim.y;

    float vertices[] = {
        -1,1,
        -1+2.0/cols,1,
        -1+2.0/cols,1-2.0/rows,
        -1,1-2.0/rows,
        -1+2.0/o_cols,1,
        -1+2.0/o_cols,1-2.0/o_rows,
        -1,1-2.0/o_rows
    } , offsets[rows*cols*2+o_rows*o_cols*2];

    for(int i = 0; i < rows; ++i){
        for(int j = 0; j < cols; ++j){
            offsets[i*cols*2+j*2  ] =  (float)j/cols*2.0;
            offsets[i*cols*2+j*2+1] = -(float)i/rows*2.0;
        }
    }

    for(int i = 0; i < o_rows; ++i){
        for(int j = 0; j < o_cols; ++j){
            offsets[rows*cols*2+i*o_cols*2+j*2  ] =  (float)j/o_cols*2.0;
            offsets[rows*cols*2+i*o_cols*2+j*2+1] = -(float)i/o_rows*2.0;
        }
    }

    unsigned int indices[] = {
        0,1,2,
        2,3,0,
        0,4,5,
        5,6,0
    };

    float *output = net.feedforward(reader.get_processed_image());

    GLuint vbo,vao,ebo,vbo_offset,vbo_color;
    glGenBuffers(1,&vbo);
    glGenBuffers(1,&vbo_offset);
    glGenBuffers(1,&vbo_color);
    glGenBuffers(1,&ebo);

    glBindBuffer(GL_ARRAY_BUFFER,vbo);
    glBufferData(GL_ARRAY_BUFFER,sizeof(vertices),vertices,GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER,vbo_offset);
    glBufferData(GL_ARRAY_BUFFER,sizeof(offsets),offsets,GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER,vbo_color);
    glBufferData(GL_ARRAY_BUFFER,sizeof(float)*rows*cols+sizeof(float)*o_cols*o_rows,NULL,GL_DYNAMIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER,0,sizeof(float)*rows*cols,reader.get_processed_image());
    glBufferSubData(GL_ARRAY_BUFFER,sizeof(float)*rows*cols,sizeof(float)*o_rows*o_cols,output);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,sizeof(indices),indices,GL_STATIC_DRAW);

    glGenVertexArrays(1,&vao);
    glBindVertexArray(vao);

    glBindBuffer(GL_ARRAY_BUFFER,vbo);
    glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,sizeof(float)*2,(void*)0);
    
    glBindBuffer(GL_ARRAY_BUFFER,vbo_offset);
    glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,sizeof(float)*2,(void*)0);

    glBindBuffer(GL_ARRAY_BUFFER,vbo_color);
    glVertexAttribPointer(2,1,GL_FLOAT,GL_FALSE,sizeof(float),(void*)0);

    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);

    glVertexAttribDivisor(1,1);
    glVertexAttribDivisor(2,1);

    SDL_Event event;
    bool quit = false, last_key = false;

    int index = 0;
    chrono::time_point<chrono::high_resolution_clock> reference = chrono::high_resolution_clock::now();
    while(!quit){
        while(SDL_PollEvent(&event)){
            if(event.type == SDL_QUIT){
                quit = true;
                break;
            }else if(event.type == SDL_KEYDOWN){
                switch(event.key.keysym.sym){
                    case SDLK_ESCAPE:
                        quit=true;
                        break;
                    case SDLK_w:
                         if(last_key) continue;
                        last_key = true;

                        if(index < sample_size-1){
                            glBindBuffer(GL_ARRAY_BUFFER,vbo_color);
                            glBufferSubData(GL_ARRAY_BUFFER,0,sizeof(float)*rows*cols,reader.get_processed_image()+784* ++index);
                            glBufferSubData(GL_ARRAY_BUFFER,sizeof(float)*rows*cols,o_cols*o_rows*sizeof(float),output+o_rows*o_cols*index);
                        }
                        break;
                    case SDLK_s:
                        if(last_key) continue;
                        last_key = true;
                        if(index > 0){
                            glBindBuffer(GL_ARRAY_BUFFER,vbo_color);
                            glBufferSubData(GL_ARRAY_BUFFER,0,sizeof(float)*rows*cols,reader.get_processed_image()+784* --index);
                            glBufferSubData(GL_ARRAY_BUFFER,sizeof(float)*rows*cols,o_cols*o_rows*sizeof(float),output+o_rows*o_cols*index);
                        }
                        break;
                }
            }else if(event.type == SDL_KEYUP){
                last_key = false;
            }
        }
        if(chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now()-reference).count()>=frame_delay){
            reference = chrono::high_resolution_clock::now();

            glClearColor(0.2f,0.2f,0.2f,1.0f);
            glClear(GL_COLOR_BUFFER_BIT);

            glUseProgram(program);
            glBindVertexArray(vao);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,ebo);

            glViewport(0,0,window_width/2,window_height);
            glDrawElementsInstanced(GL_TRIANGLES,6,GL_UNSIGNED_INT,(void*)0,cols*rows);

            glViewport(window_width/2,0,window_width/2,window_height);
            glDrawElementsInstancedBaseInstance(GL_TRIANGLES,6,GL_UNSIGNED_INT,(void*)(sizeof(int)*6),o_rows*o_cols,rows*cols);
        
            SDL_GL_SwapWindow(window);
        }
    }

    SDL_GL_DeleteContext(context); 
    SDL_Quit();
    return 0;
}