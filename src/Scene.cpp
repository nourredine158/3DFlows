#include "Scene.h"

#include <cstdint>

#include "bgfx/bgfx.h"
#include "bx/bx.h"
#include "bx/macros.h"
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>


#include "bx/math.h"
#include "bx/timer.h"

#include "Particle.h"
#include <glm/geometric.hpp>



uint32_t sphereColor = 0x94e24d;
uint32_t boxColor = 0x94e24d;

float gravity = 0.0f;
float particleRadius = 0.5f;
int nbParticles = 40;
float collisionDamping = 0.8f;

struct PosColorVertex
{
	float x;
	float y;
	float z;
	uint32_t abgr;
};

static PosColorVertex cubeVertices[] =
{
	{-1.0f,  1.0f,  1.0f, 0xff000000 },
	{ 1.0f,  1.0f,  1.0f, 0xff0000ff },
	{-1.0f, -1.0f,  1.0f, 0xff00ff00 },
	{ 1.0f, -1.0f,  1.0f, 0xff00ffff },
	{-1.0f,  1.0f, -1.0f, 0xffff0000 },
	{ 1.0f,  1.0f, -1.0f, 0xffff00ff },
	{-1.0f, -1.0f, -1.0f, 0xffffff00 },
	{ 1.0f, -1.0f, -1.0f, 0xffffffff },
};

static const uint16_t cubeTriList[] =
{
	0, 1, 2,
	1, 3, 2,
	4, 6, 5,
	5, 6, 7,
	0, 2, 4,
	4, 2, 6,
	1, 5, 3,
	5, 7, 3,
	0, 4, 1,
	4, 5, 1,
	2, 3, 6,
	6, 3, 7,
};

static const char* s_ptNames[]
{
	"Triangle List",
	"Triangle Strip",
	"Lines",
	"Line Strip",
	"Points",
};

static const uint64_t s_ptState[]
{
	UINT64_C(0),
	BGFX_STATE_PT_TRISTRIP,
	BGFX_STATE_PT_LINES,
	BGFX_STATE_PT_LINESTRIP,
	BGFX_STATE_PT_POINTS,
};
BX_STATIC_ASSERT(BX_COUNTOF(s_ptState) == BX_COUNTOF(s_ptNames));

// Function to generate vertices for a sphere
void GenerateSphereVertices(PosColorVertex* vertices, float radius, int slices, int stacks)
{
	const float pi = 3.14159265359f;

	int vertexIndex = 0;
	for (int stack = 0; stack <= stacks; ++stack)
	{
		float phi = pi * stack / stacks;
		float cosPhi = std::cos(phi);
		float sinPhi = std::sin(phi);

		for (int slice = 0; slice < slices; ++slice)
		{
			float theta = 2 * pi * slice / slices;
			float cosTheta = std::cos(theta);
			float sinTheta = std::sin(theta);

			vertices[vertexIndex].x = radius * sinPhi * cosTheta; // x
			vertices[vertexIndex].y = radius * sinPhi * sinTheta; // y
			vertices[vertexIndex].z = radius * cosPhi;            // z
			vertices[vertexIndex].abgr = sphereColor;              // white color

			++vertexIndex;
		}
	}

}

// Function to generate indices for a sphere
void GenerateSphereIndices(uint16_t* indices, int slices, int stacks)
{
	int index = 0;
	for (int stack = 0; stack < stacks; ++stack)
	{
		int stackStart = stack * slices;
		int nextStackStart = (stack + 1) * slices;

		for (int slice = 0; slice < slices; ++slice)
		{
			// First triangle of quad
			indices[index++] = stackStart + slice;
			indices[index++] = stackStart + (slice + 1) % slices;
			indices[index++] = nextStackStart + slice;

			// Second triangle of quad
			indices[index++] = nextStackStart + slice;
			indices[index++] = stackStart + (slice + 1) % slices;
			indices[index++] = nextStackStart + (slice + 1) % slices;
		}
	}
}

void GenerateBox(PosColorVertex* vertices, uint16_t* indices, float boxWidth, float boxHeight)
{
	vertices[0] = { -boxWidth / 2, -boxHeight / 2, 0, boxColor };
	vertices[1] = { boxWidth / 2, -boxHeight / 2, 0, boxColor };
	vertices[2] = { boxWidth / 2, boxHeight / 2, 0, boxColor };
	vertices[3] = { -boxWidth / 2, boxHeight / 2, 0, boxColor };

	indices[0] = 0;
	indices[1] = 1;
	indices[2] = 2;
	indices[3] = 3;
	indices[4] = 0;
}

namespace scene
{
	PosColorVertex* sphereVertices = nullptr;
	uint16_t* sphereIndices = nullptr;

	bgfx::VertexBufferHandle m_vbh_sphere;
	bgfx::IndexBufferHandle m_ibh_sphere;

	float boxWidth = 25;
	float boxHeight = 15;

	PosColorVertex* boxVertices = nullptr;
	uint16_t* boxIndices = nullptr;

	bgfx::VertexBufferHandle m_vbh_box;
	bgfx::IndexBufferHandle m_ibh_box;

	bgfx::ProgramHandle m_program;

	int64_t m_timeOffset;

	std::vector<Particle> particles(nbParticles);
}

bgfx::ShaderHandle scene::loadShader(const char* FILENAME)
{
	const char* shaderPath = "???";

	switch (bgfx::getRendererType())
	{
	case bgfx::RendererType::Noop:
	case bgfx::RendererType::Direct3D11:
	case bgfx::RendererType::Direct3D12: shaderPath = "shaders/dx11/";  break;
	case bgfx::RendererType::Agc:
	case bgfx::RendererType::Gnm:        shaderPath = "shaders/pssl/";  break;
	case bgfx::RendererType::Metal:      shaderPath = "shaders/metal/"; break;
	case bgfx::RendererType::Nvn:        shaderPath = "shaders/nvn/";   break;
	case bgfx::RendererType::OpenGL:     shaderPath = "shaders/glsl/";  break;
	case bgfx::RendererType::OpenGLES:   shaderPath = "shaders/essl/";  break;
	case bgfx::RendererType::Vulkan:     shaderPath = "shaders/spirv/"; break;

	case bgfx::RendererType::Count:
		BX_ASSERT(false, "You should not be here!");
		break;
	}

	size_t shaderLen = strlen(shaderPath);
	size_t fileLen = strlen(FILENAME);

	char* filePath = (char*)calloc(1, shaderLen + fileLen + 1);

	memcpy(filePath, shaderPath, shaderLen);
	memcpy(&filePath[shaderLen], FILENAME, fileLen);


	FILE* file = fopen(filePath, "rb");
	fseek(file, 0, SEEK_END);
	long fileSize = ftell(file);
	fseek(file, 0, SEEK_SET);

	const bgfx::Memory* mem = bgfx::alloc(fileSize + 1);
	fread(mem->data, 1, fileSize, file);
	mem->data[mem->size - 1] = '\0';
	fclose(file);

	return bgfx::createShader(mem);
}

float getRandomfloat()
{
	std::random_device rd;  // Obtain a random number from hardware
	std::mt19937 gen(rd()); // Seed the generator
	std::uniform_real_distribution<> dis(-1.0, 1.0); // Define the range

	// Generate a random number between -1 and 1
	return float(dis(gen));
}

void initParticles(std::vector<Particle>& particles)
{
	float force = 10;
	for (auto& particle : particles)
	{
		float x = getRandomfloat();
		float y = getRandomfloat();

		glm::vec3 vol = { x , y , 0.0f };

		particle.velocity = glm::normalize(vol) * force;
		particle.radius = particleRadius;
	}
}

void scene::init()
{

	bgfx::VertexLayout pcvDecl;
	pcvDecl.begin()
		.add(bgfx::Attrib::Position, 3, bgfx::AttribType::Float)
		.add(bgfx::Attrib::Color0, 4, bgfx::AttribType::Uint8, true)
		.end();

	const int slices = 20; // Increase for smoother sphere
	const int stacks = 20; // Increase for more detailed sphere
	const float radius = particleRadius;

	const int numVertices = slices * (stacks + 1);
	const int numIndices = slices* stacks * 6;


	sphereVertices = new PosColorVertex[numVertices];
	GenerateSphereVertices(sphereVertices, radius, slices, stacks);

	sphereIndices = new uint16_t[numIndices];
	GenerateSphereIndices(sphereIndices, slices, stacks);

	m_vbh_sphere = bgfx::createVertexBuffer(bgfx::makeRef(sphereVertices, sizeof(PosColorVertex) * numVertices), pcvDecl);
	m_ibh_sphere = bgfx::createIndexBuffer(bgfx::makeRef(sphereIndices, sizeof(uint16_t) * numIndices));

	boxVertices = new PosColorVertex[4];
	boxIndices = new uint16_t[5];
	GenerateBox(boxVertices, boxIndices, boxWidth, boxHeight);

	m_vbh_box = bgfx::createVertexBuffer(bgfx::makeRef(boxVertices, sizeof(PosColorVertex) * 4), pcvDecl);
	m_ibh_box = bgfx::createIndexBuffer(bgfx::makeRef(boxIndices, sizeof(uint16_t) * 5));

	bgfx::ShaderHandle vsh = loadShader("vs_cubes.bin");
	bgfx::ShaderHandle fsh = loadShader("fs_cubes.bin");

	m_program = bgfx::createProgram(vsh, fsh, true);

	m_timeOffset = bx::getHPCounter();

	initParticles(particles);
}


void resolveBoxCollisions(Particle& particle)
{
	float xBound = scene::boxWidth / 2 - particle.radius;
	if(abs(particle.position.x) > xBound)
	{
		particle.position.x = xBound * glm::sign(particle.position.x);
		particle.velocity.x *= -1 * collisionDamping;
	}

	float yBound = scene::boxHeight / 2 - particle.radius;
	if (abs(particle.position.y) > yBound)
	{
		particle.position.y = yBound * glm::sign(particle.position.y);
		particle.velocity.y *= -1 * collisionDamping;
	}
}

void scene::update(int width, int height)
{
	//float time = (float)((bx::getHPCounter() - m_timeOffset) / double(bx::getHPFrequency()));
	float time = 0.01667f;

	// view
	{
		const bx::Vec3 at = { 0.0f, 0.0f,  0.0f };
		const bx::Vec3 eye = { 0.0f, 0.0f, -5.0f };
		float view[16];
		bx::mtxLookAt(view, eye, at);
		float proj[16];
		//bx::mtxProj(proj, 60.0f, float(width) / float(height), 0.1f, 100.0f, bgfx::getCaps()->homogeneousDepth);

		
		// Calculate the aspect ratio
		float aspectRatio = float(width) / float(height);

		// Define the parameters for the orthographic projection
		float orthoSize = 10.0f; // Size of the orthographic projection
		float nearPlane = 0.1f; // Near clipping plane
		float farPlane = 100.0f; // Far clipping plane

		// Generate the orthographic projection matrix
		bx::mtxOrtho(proj, -orthoSize * aspectRatio, orthoSize * aspectRatio, -orthoSize, orthoSize, nearPlane, farPlane, 0, bgfx::getCaps()->homogeneousDepth);
		
		bgfx::setViewTransform(0, view, proj);
	}
	
	// Sphere
	for(auto& particle : particles)
	{
		uint64_t state = 0
			| BGFX_STATE_WRITE_R
			| BGFX_STATE_WRITE_G
			| BGFX_STATE_WRITE_B
			| BGFX_STATE_WRITE_A
			| BGFX_STATE_WRITE_Z
			| BGFX_STATE_DEPTH_TEST_LESS
			| BGFX_STATE_CULL_CW
			| BGFX_STATE_MSAA
			| s_ptState[0]
			;

		glm::vec3 down = { 0.0f, -1.0f, 0.0f };

		particle.velocity += down * gravity * time;
		particle.position += particle.velocity * time;

		resolveBoxCollisions(particle);

		float mtx[16];
		bx::mtxTranslate(mtx, particle.position.x, particle.position.y, particle.position.z);
		bgfx::setTransform(mtx);

		bgfx::setVertexBuffer(0, m_vbh_sphere);
		bgfx::setIndexBuffer(m_ibh_sphere);

		// Set render states.
		bgfx::setState(state);

		bgfx::submit(0, m_program);
	}

	// Box
	{
		uint64_t state = 0
			| BGFX_STATE_WRITE_R
			| BGFX_STATE_WRITE_G
			| BGFX_STATE_WRITE_B
			| BGFX_STATE_WRITE_A
			| BGFX_STATE_WRITE_Z
			| BGFX_STATE_DEPTH_TEST_LESS
			| BGFX_STATE_CULL_CW
			| BGFX_STATE_MSAA
			| s_ptState[3]
			;

		bgfx::setVertexBuffer(0, m_vbh_box);
		bgfx::setIndexBuffer(m_ibh_box);

		// Set render states.
		bgfx::setState(state);

		bgfx::submit(0, m_program);
	}

}

void scene::shutdown()
{
	bgfx::destroy(m_ibh_sphere);
	bgfx::destroy(m_vbh_sphere);

	delete[] sphereVertices;
	delete[] sphereIndices;

	bgfx::destroy(m_ibh_box);
	bgfx::destroy(m_vbh_box);

	delete[] boxVertices;
	delete[] boxIndices;

	bgfx::destroy(m_program);
}