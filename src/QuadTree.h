#pragma once

#include "glm/vec2.hpp"
#include "glm/vec4.hpp"

// Node class for the Quadtree
class QuadtreeNode {
public:
    glm::vec4 boundary;
    std::vector<Particle*> particles;
    QuadtreeNode* children[4];
    int capacity;
    bool isSubdivided = false;

    QuadtreeNode(const glm::vec4& _boundary, const int capacity) : boundary(_boundary), capacity(capacity)
	{
        for (int i = 0; i < 4; ++i)
            children[i] = nullptr;
    }

    ~QuadtreeNode()
	{
        for (int i = 0; i < 4; ++i) {
            if (children[i] != nullptr) {
                delete children[i];
                children[i] = nullptr;
            }
        }
    }

    bool contains(const glm::vec2& point) const {
        return (point.x >= boundary.x && point.x <= boundary.x + boundary.z && point.y >= boundary.y && point.y <= boundary.y + boundary.w);
    }

    bool intersectWithCircle(glm::vec2& circlePosition, float radius)
	{
        // temporary variables to set edges for testing
        float testX = circlePosition.x;
        float testY = circlePosition.y;

        // which edge is closest?
        if (circlePosition.x < boundary.x)         testX = boundary.x;      // test left edge
        else if (circlePosition.x > boundary.x + boundary.z) testX = boundary.x + boundary.z;   // right edge
        if (circlePosition.y < boundary.y)         testY = boundary.y;      // top edge
        else if (circlePosition.y > boundary.y + boundary.w) testY = boundary.y + boundary.w;   // bottom edge

        // get distance from closest edges
        float distX = circlePosition.x - testX;
        float distY = circlePosition.y - testY;
        float distance = sqrt((distX * distX) + (distY * distY));

        // if the distance is less than the radius, collision!
        if (distance <= radius) {
            return true;
        }
        return false;
    }
};


// Quadtree class
class Quadtree
{
public:
    QuadtreeNode* root;

    Quadtree(const glm::vec4& boundary, int capacity) {
        root = new QuadtreeNode(boundary, capacity);
    }

    ~Quadtree() {
        delete root;
    }

    // Insert a point into the quadtree
    void insert(Particle& particle) {
        insertHelper(particle, root);
    }

    // Helper function to insert a point recursively
    void insertHelper(Particle& particle, QuadtreeNode* node) {
        if (!node->contains(particle.position))
            return;

        if (!node->isSubdivided)
        {
            if (node->particles.size() < node->capacity)
            {
                node->particles.push_back(&particle);
                return;
            }

        	subdivideNode(node);
        }


        for (auto i : node->children)
        {
            insertHelper(particle, i);
        }
    }

    std::vector<Particle*> findParticles(glm::vec2& circlePosition, float radius)
    {
        return findParticles(circlePosition, radius, root);
    }

    std::vector<Particle*> findParticles(glm::vec2& circlePosition, float radius, QuadtreeNode* node)
    {
        std::vector<Particle*> found = {};
        if (!node->intersectWithCircle(circlePosition, radius))
	        return found;

        if(!node->isSubdivided)
        {
        	for (auto particle : node->particles)
        	{
                glm::vec2 particlePos(particle->position);
        		if (glm::distance(particlePos, circlePosition) <= radius * 2.f)
					found.emplace_back(particle);
        	}

        }
        else
        {
            for (auto& i : node->children)
            {
                std::vector<Particle*> foundNew = findParticles(circlePosition, radius, i);
                found.insert(found.end(), foundNew.begin(), foundNew.end());
            }
                
        }

        return found;
    }

    // Subdivide a node into four children
    void subdivideNode(QuadtreeNode* node)
	{
        float x = node->boundary.x;
        float y = node->boundary.y;
        float w = node->boundary.z;
        float h = node->boundary.w;

        glm::vec4 nw(x, y, w / 2, h / 2);
        glm::vec4 ne(x + w / 2, y, w / 2, h / 2);
        glm::vec4 sw(x, y + h / 2, w / 2, h / 2);
        glm::vec4 se(x + w / 2, y + h / 2, w / 2, h / 2);

        node->children[0] = new QuadtreeNode(nw, node->capacity);
        node->children[1] = new QuadtreeNode(ne, node->capacity);
        node->children[2] = new QuadtreeNode(sw, node->capacity);
        node->children[3] = new QuadtreeNode(se, node->capacity);

        // Distribute points to children
        for (auto paticle : node->particles) 
        {
            for (int i = 0; i < 4; ++i) 
            {
                if (node->children[i]->contains(paticle->position))
                {
                    insertHelper(*paticle, node->children[i]);
                    break;
                }
            }
        }
        node->particles.clear();

        node->isSubdivided = true;
    }
};