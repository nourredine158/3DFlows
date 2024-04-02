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

#include "glm/ext/scalar_constants.hpp"
#include <imgui.h>
#include <iostream>

#include "QuadTree.h"
#include <glm/ext/matrix_float4x4.hpp>

struct PosColorVertex
{
	float x;
	float y;
	float z;
	uint32_t abgr;
};

int pixelWidth = 0;
int pixelHeight = 0;

int xMousePos = 0;
int yMousePos = 0;

PosColorVertex* sphereVertices = nullptr;
uint16_t* sphereIndices = nullptr;

bgfx::VertexBufferHandle m_vbh_sphere;
bgfx::IndexBufferHandle m_ibh_sphere;

float boxWidth = 20;
float boxHeight = 10;

std::vector<PosColorVertex> boxVertices;
std::vector<uint16_t> boxIndices;

bgfx::VertexBufferHandle m_vbh_box;
bgfx::IndexBufferHandle m_ibh_box;

bgfx::ProgramHandle m_program;

int64_t m_timeOffset;

int nbParticles = 200;
std::vector<Particle> particles(nbParticles);

glm::vec4 boundary(-boxWidth / 2.f, -boxHeight / 2.f, boxWidth, boxHeight); // Example boundary

int quadCapacity = 500;



uint32_t sphereColor = 0x4eb3ff;
uint32_t boxColor = 0x94e24d;

float gravity = 0.0f;
float particleRadius = 0.1f;
float smoothingRadius = 1.5f;


float targetDensity =0.5f;
float pressureMultiplier = 0.5f;

float collisionDamping = 0.8f;





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

void GenerateBox(std::vector<PosColorVertex>& vertices, std::vector<uint16_t>& indices, float boxWidth, float boxHeight)
{
	std::vector<PosColorVertex> newVertices =
	{
		{ -boxWidth / 2, -boxHeight / 2, 0, boxColor },
		{ boxWidth / 2, -boxHeight / 2, 0, boxColor },
		{ boxWidth / 2, boxHeight / 2, 0, boxColor },
		{ -boxWidth / 2, boxHeight / 2, 0, boxColor }
	};

	vertices.insert(vertices.begin(), newVertices.begin(), newVertices.end());

	std::vector<uint16_t> newIndices =
	{
		0,
		1,
		1,
		2,
		2,
		3,
		3,
		0
	};

	indices.insert(indices.begin(), newIndices.begin(), newIndices.end());

}

namespace scene
{

	
}

void scene::updateMousePos(int xPos, int yPos)
{
	xMousePos = xPos;
	yMousePos = yPos;
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
	float width = 20.f * particleRadius * 2;
	float height = 10.f * particleRadius * 2;

	float pointerY = -height / 2.f;
	float pointerX = -width / 2.f;

	float force = 10;
	int nbParticle = 0;
	for (auto& particle : particles)
	{
		if (nbParticle % 10 == 0)
		{
			pointerY += particleRadius * 2;
			pointerX = -width / 2.f;
		}


		float x = getRandomfloat();
		float y = getRandomfloat();

		glm::vec3 vol = { x , y , 0.0f };

		//particle.velocity = glm::normalize(vol) * force;

		float xPosition = boxWidth / 2.f * x - particleRadius;
		float yPosition = boxHeight / 2.f * y - particleRadius;

		particle.position = { xPosition , yPosition, 0.f };
		particle.radius = particleRadius;

		//scene::quadtree.insert({ xPosition, yPosition });

		pointerX += particleRadius * 2;
		nbParticle++;
	}
}

void browse(std::vector<PosColorVertex>& vertices, std::vector<uint16_t>& indices, QuadtreeNode* node) {
	if (node == nullptr)
		return;

	
	if(node->isSubdivided)
	{
		std::vector<PosColorVertex> newVertices =
		{
			{node->boundary.x + node->boundary.z / 2.f, node->boundary.y, 0.f, boxColor},
			{node->boundary.x + node->boundary.z / 2.f, node->boundary.y + node->boundary.w, 0.f, boxColor},
			{node->boundary.x, node->boundary.y + node->boundary.w / 2.f, 0.f, boxColor},
			{node->boundary.x + node->boundary.z, node->boundary.y + node->boundary.w / 2.f, 0.f, boxColor},
		};

		uint16_t i1 = vertices.size();
		uint16_t i2 = i1 + 1;
		uint16_t i3 = i1 + 2;
		uint16_t i4 = i1 + 3;

		std::vector<uint16_t> newIndices =
		{
			i1, i2, i3, i4
		};

		vertices.insert(vertices.end(), newVertices.begin(), newVertices.end());
		indices.insert(indices.end(), newIndices.begin(), newIndices.end());
	}

	// Recursively browse children
	for (int i = 0; i < 4; ++i) {
		browse(vertices, indices, node->children[i]);
	}
}

void scene::init()
{
	initParticles(particles);


	bgfx::VertexLayout pcvDecl;
	pcvDecl.begin()
		.add(bgfx::Attrib::Position, 3, bgfx::AttribType::Float)
		.add(bgfx::Attrib::Color0, 4, bgfx::AttribType::Uint8, true)
		.end();

	const int slices = 5; // Increase for smoother sphere
	const int stacks = 5; // Increase for more detailed sphere
	const float radius = particleRadius;

	const int numVertices = slices * (stacks + 1);
	const int numIndices = slices* stacks * 6;


	sphereVertices = new PosColorVertex[numVertices];
	GenerateSphereVertices(sphereVertices, radius, slices, stacks);

	sphereIndices = new uint16_t[numIndices];
	GenerateSphereIndices(sphereIndices, slices, stacks);

	m_vbh_sphere = bgfx::createVertexBuffer(bgfx::makeRef(sphereVertices, sizeof(PosColorVertex) * numVertices), pcvDecl);
	m_ibh_sphere = bgfx::createIndexBuffer(bgfx::makeRef(sphereIndices, sizeof(uint16_t) * numIndices));


	GenerateBox(boxVertices, boxIndices, boxWidth, boxHeight);
	//browse(boxVertices, boxIndices, scene::quadtree.root);

	m_vbh_box = bgfx::createVertexBuffer(bgfx::makeRef(boxVertices.data(), boxVertices.size() * sizeof(PosColorVertex)), pcvDecl);
	m_ibh_box = bgfx::createIndexBuffer(bgfx::makeRef(boxIndices.data(), boxIndices.size() * sizeof(uint16_t)));

	bgfx::ShaderHandle vsh = loadShader("vs_cubes.bin");
	bgfx::ShaderHandle fsh = loadShader("fs_cubes.bin");

	m_program = bgfx::createProgram(vsh, fsh, true);

	m_timeOffset = bx::getHPCounter();

	
}


void resolveBoxCollisions(Particle& particle)
{
	float xBound = boxWidth / 2 - particle.radius;
	if(abs(particle.position.x) > xBound)
	{
		particle.position.x = xBound * glm::sign(particle.position.x);
		particle.velocity.x *= -1 * collisionDamping;
	}

	float yBound = boxHeight / 2 - particle.radius;
	if (abs(particle.position.y) > yBound)
	{
		particle.position.y = yBound * glm::sign(particle.position.y);
		particle.velocity.y *= -1 * collisionDamping;
	}
}

float smoothingKernel(float radius, float dst)
{
	if (dst >= radius) return 0.f;

	float volume = (glm::pi<float>() * glm::pow(radius, 4.0f)) / 6.0f;
	return (radius - dst) * (radius - dst) / volume;
}

float smoothingKernelDerivative(float radius, float dst)
{
	if (dst >= radius) return 0.f;

	float scale = 12.f / (glm::pi<float>() * glm::pow(radius, 4.f));

	return scale * (dst - radius);
}

float calculateDensity(glm::vec3 samplePoint, Quadtree& quadtree)
{
	float density = 0.0f;
	const float mass = 1.0f;

	glm::vec2 point = samplePoint;
	std::vector<Particle*> found = quadtree.findParticles(point, smoothingRadius);

	for (auto particle : found)
	{
		float dst = glm::length(particle->position - samplePoint);
		float influence = smoothingKernel(smoothingRadius, dst);
		density += mass * influence;
	}

	return density;
}

float convertDensityToPressure(float density)
{
	float densityError = density - targetDensity;
	float pressure = densityError * pressureMultiplier;
	return pressure;
}

float calculateSharedPressure(float densityA, float densityB)
{
	float pressureA = convertDensityToPressure(densityA);
	float pressureB = convertDensityToPressure(densityB);

	return (pressureA + pressureB) / 2.0f;
}

glm::vec3 calculatePressureForce(Particle& particleIn, Quadtree& quadtree)
{
	glm::vec3 pressureForce = {};

	const float mass = 1.0f;

	glm::vec2 point = particleIn.position;
	std::vector<Particle*> found = quadtree.findParticles(point, smoothingRadius);
	for (auto particle : found)
	{
		if(particle == &particleIn)
			continue;
		float dst = glm::length(particle->position - particleIn.position);

		glm::vec3 dir = {};
		if(dst == 0.0f)
		{
			float x = getRandomfloat();
			float y = getRandomfloat();

			glm::vec3 vol = { x , y , 0.0f };

			dir = glm::normalize(vol);
		}
		else
		{
			dir = (particle->position - particleIn.position) / dst;
		}
		
		float slope = smoothingKernelDerivative(smoothingRadius, dst);
		float density = particle->density;
		float sharedPressure = calculateSharedPressure(density, particleIn.density);
		pressureForce += sharedPressure * dir * slope * mass / density;
	}

	return pressureForce;
}

void updateSpatialLookup(float radius)
{
	
}

glm::vec3 viewportToWorld(float xMousePos, float yMousePos, float viewportWidth, float viewportHeight,
	const glm::mat4& projectionMatrix, const glm::mat4& viewMatrix) {
	// Convert viewport coordinates to normalized device coordinates
	float ndcX = (2.0f * xMousePos) / viewportWidth - 1.0f;
	float ndcY = 1.0f - (2.0f * yMousePos) / viewportHeight;

	// Create a 4D vector with the x and y coordinates in NDC, z = 0 for near plane, w = 1
	glm::vec4 viewportPos(ndcX, ndcY, 0.0f, 1.0f);

	// Invert the view-projection matrix to get from clip space to eye space
	glm::mat4 invVP = glm::inverse(projectionMatrix * viewMatrix);

	// Transform the point from clip space to eye space
	glm::vec4 eyePos = invVP * viewportPos;

	// Convert from homogeneous coordinates to 3D space
	glm::vec3 worldPos = glm::vec3(eyePos) / eyePos.w;

	return worldPos;
}


void scene::update(int width, int height)
{
	pixelWidth = width;
	pixelHeight = height;

	Quadtree quadtree(boundary, quadCapacity);
	

	for (auto& particle : particles)
	{
		quadtree.insert(particle);
	}

	const auto& io = ImGui::GetIO();
	float framerate = 1.0f / io.DeltaTime;

	// ImGui
	{
		ImGui::Begin("Float Value Editor");

		// Display the current value of the float
		ImGui::Text("frameRate: %.3f", framerate);

		// Display the current value of the float
		ImGui::Text("pressureMultiplier: %.3f", pressureMultiplier);

		// Input field to modify the float value
		ImGui::InputFloat("##floatInput", &pressureMultiplier);

		// Display the current value of the float
		ImGui::Text("targetDensity: %.3f", targetDensity);

		// Input field to modify the float value
		ImGui::InputFloat("##targetDensityInput", &targetDensity);

		// Display the current value of the float
		ImGui::Text("smoothingRadius: %.3f", smoothingRadius);

		// Input field to modify the float value
		ImGui::InputFloat("##smoothingRadiusInput", &smoothingRadius);

		// Display the current value of the float
		ImGui::Text("gravity: %.3f", gravity);

		// Input field to modify the float value
		ImGui::InputFloat("##gravityInput", &gravity);

		ImGui::End();
	}
	//float time = (float)((bx::getHPCounter() - m_timeOffset) / double(bx::getHPFrequency()));
	float time = io.DeltaTime;


	// view
	
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

	glm::mat4 projectionMatrix = glm::mat4(
		proj[0], proj[1], proj[2], proj[3],
		proj[4], proj[5], proj[6], proj[7],
		proj[8], proj[9], proj[10], proj[11],
		proj[12], proj[13], proj[14], proj[15]
	);

	glm::mat4 viewMatrix = glm::mat4(
		view[0], view[1], view[2], view[3],
		view[4], view[5], view[6], view[7],
		view[8], view[9], view[10], view[11],
		view[12], view[13], view[14], view[15]
	);


	glm::vec3 world = viewportToWorld(xMousePos, yMousePos, width, height, projectionMatrix, viewMatrix);

	//std::cout << "X : " << world.x << ", Y : " << world.y << ", Z : " << world.z << "\n";

	//glm::vec2 mouseWorld(world.x, world.y);



	// gravity and density
	
	for (auto& particle : particles)
	{


		particle.predictedPosition = particle.position + particle.velocity * 1.f / 120.f;
	}

	

	//std::cout << "==> : " << found.size() << "\n";

	
	// densities
	for (auto& particle : particles)
	{
		particle.density = calculateDensity(particle.predictedPosition, quadtree);
	}
	
	for (auto& particle : particles)
	{
		glm::vec3 pressureForce = calculatePressureForce(particle, quadtree);
		glm::vec3 pressureAcceleration = pressureForce / particle.density;

		particle.velocity = pressureAcceleration * time;

		glm::vec3 down = { 0.0f, -1.0f, 0.0f };
		particle.velocity += down * gravity * time;
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
			| s_ptState[2]
			;

		bgfx::VertexLayout pcvDecl;
		pcvDecl.begin()
			.add(bgfx::Attrib::Position, 3, bgfx::AttribType::Float)
			.add(bgfx::Attrib::Color0, 4, bgfx::AttribType::Uint8, true)
			.end();




		boxVertices.clear();
		boxIndices.clear();

		GenerateBox(boxVertices, boxIndices, boxWidth, boxHeight);
		browse(boxVertices, boxIndices, quadtree.root);


		// Only for debug
		bgfx::destroy(m_ibh_box);
		bgfx::destroy(m_vbh_box);
		
		m_vbh_box = bgfx::createVertexBuffer(bgfx::makeRef(boxVertices.data(), boxVertices.size() * sizeof(PosColorVertex)), pcvDecl);
		m_ibh_box = bgfx::createIndexBuffer(bgfx::makeRef(boxIndices.data(), boxIndices.size() * sizeof(uint16_t)));

		//
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

	bgfx::destroy(m_program);
}