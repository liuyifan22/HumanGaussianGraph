import numpy as np
from OpenGL.GL import *
import glfw

# Initialize GLFW
if not glfw.init():
    raise Exception("GLFW can't be initialized")

# Create a windowed mode window and its OpenGL context
window = glfw.create_window(640, 480, "OpenGL Context", None, None)
if not window:
    glfw.terminate()
    raise Exception("GLFW window can't be created")

# Make the window's context current
glfw.make_context_current(window)

vertex_shader_code = """
#version 130
in vec3 position;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
flat out int face_id;
void main()
{
    gl_Position = projection * view * model * vec4(position, 1.0);
    face_id = gl_VertexID / 3;  // Assuming each face is a triangle
}
"""

fragment_shader_code = """
#version 130
flat in int face_id;
out vec4 FragColor;
void main()
{
    int r = (face_id & 0xFF0000) >> 16;
    int g = (face_id & 0x00FF00) >> 8;
    int b = (face_id & 0x0000FF);
    assert(r==0);
    FragColor = vec4(float(r) / 255.0, float(g) / 255.0, float(b) / 255.0, 1.0);
}
"""

def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
        raise RuntimeError(glGetShaderInfoLog(shader))
    return shader

def create_shader_program(vertex_code, fragment_code):
    vertex_shader = compile_shader(vertex_code, GL_VERTEX_SHADER)
    fragment_shader = compile_shader(fragment_code, GL_FRAGMENT_SHADER)
    program = glCreateProgram()
    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)
    glLinkProgram(program)
    if glGetProgramiv(program, GL_LINK_STATUS) != GL_TRUE:
        print("Vertex Shader Log:", glGetShaderInfoLog(vertex_shader))
        print("Fragment Shader Log:", glGetShaderInfoLog(fragment_shader))
        print("Program Log:", glGetProgramInfoLog(program))
        raise RuntimeError(glGetProgramInfoLog(program))
    return program

shader_program = create_shader_program(vertex_shader_code, fragment_shader_code)

# Clean up and terminate GLFW
glfw.terminate()

# # filepath: ./smpl_renderer/mvhuman_tools/visual_smpl/mytools/face_index_shader.py
# import numpy as np
# from OpenGL.GL import *

# vertex_shader_code = """
# #version 330 core
# layout(location = 0) in vec3 position;
# uniform mat4 model;
# uniform mat4 view;
# uniform mat4 projection;
# flat out int face_id;
# void main()
# {
#     gl_Position = projection * view * model * vec4(position, 1.0);
#     face_id = gl_VertexID / 3;  // Assuming each face is a triangle
# }
# """

# fragment_shader_code = """
# #version 330 core
# flat in int face_id;
# out vec4 FragColor;
# void main()
# {
#     FragColor = vec4(float(face_id) / 255.0, 0.0, 0.0, 1.0);  // Encode face_id as red channel
# }
# """

# def compile_shader(source, shader_type):
#     shader = glCreateShader(shader_type)
#     glShaderSource(shader, source)
#     glCompileShader(shader)
#     if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
#         raise RuntimeError(glGetShaderInfoLog(shader))
#     return shader

# def create_shader_program(vertex_code, fragment_code):
#     vertex_shader = compile_shader(vertex_code, GL_VERTEX_SHADER)
#     fragment_shader = compile_shader(fragment_code, GL_FRAGMENT_SHADER)
#     program = glCreateProgram()
#     glAttachShader(program, vertex_shader)
#     glAttachShader(program, fragment_shader)
#     glLinkProgram(program)
#     if glGetProgramiv(program, GL_LINK_STATUS) != GL_TRUE:
#         raise RuntimeError(glGetProgramInfoLog(program))
#     return program

# shader_program = create_shader_program(vertex_shader_code, fragment_shader_code)