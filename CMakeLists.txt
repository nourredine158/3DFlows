# CMakeList.txt : projet CMake pour 3DFlows, incluez la source et définissez
# la logique spécifique au projet ici.
#
cmake_minimum_required (VERSION 3.8)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Activez Rechargement à chaud pour les compilateurs MSVC si cela est pris en charge.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()


project ("3DFlows")

# Add GLFW subdirectory
add_subdirectory(Dependencies/glfw)

# Add GLM subdirectory
add_subdirectory(Dependencies/glm)

# Add BGFX subdirectory
add_subdirectory(Dependencies/bgfx.cmake)

# Include directories for GLFW, GLM, and BGFX
include_directories(Dependencies/glfw/include)
include_directories(Dependencies/glm)
include_directories(Dependencies/bgfx.cmake/bgfx/include)
include_directories(Dependencies/bgfx.cmake/bgfx/examples/common)
include_directories(Dependencies/bgfx.cmake/bx/include)
include_directories(Dependencies/bgfx.cmake/bx/include/compat/msvc)
include_directories(Dependencies/bgfx.cmake/bimg/include)

# Add your source files here
# Gather all source and header files in the src directory
file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.h")

# Ajoutez une source à l'exécutable de ce projet.
add_executable (${PROJECT_NAME} ${SOURCES})

target_compile_options(${PROJECT_NAME} PUBLIC "/Zc:__cplusplus")

# Link GLFW library
target_link_libraries(${PROJECT_NAME} glfw)

# Link GLFW library
target_link_libraries(${PROJECT_NAME} glm)

# Link BGFX library
target_link_libraries(${PROJECT_NAME} bgfx)

set(BGFX_BUILD_TOOLS_SHADER ON)