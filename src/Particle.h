#pragma once
#include <glm/vec3.hpp>

class Particle
{
public:
	float radius = 1.0;

	glm::vec3 position = {};
	glm::vec3 velocity = {};
};