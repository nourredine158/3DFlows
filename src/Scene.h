#pragma once
#include <bgfx/bgfx.h>


namespace scene
{
	void init();

	void update(int width, int height);

	void shutdown();

	bgfx::ShaderHandle loadShader(const char* FILENAME);
}
