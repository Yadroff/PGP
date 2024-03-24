#include <fstream>

#include "object.h"

#include <memory>

#include "Common/common_structures.cuh"
#include "Math/math.cuh"
#include "Primitives/circle.h"

engOBJECT& engOBJECT::Move(const vec3& pos)
{
	for (auto& vert : vertices)
	{
		vert += pos;
	}
	return *this;
}

engOBJECT& engOBJECT::Rotate(const vec3& rot)
{
	float sinX = sin(rot.x), sinY = sin(rot.y), sinZ = sin(rot.z);
	float cosX = cos(rot.x), cosY = cos(rot.y), cosZ = cos(rot.z);

	mat3 rotateX{
		1.0, 0.0, 0.0,
		0.0, cosX, -sinX,
		0.0, sinX, cosX
	};
	mat3 rotateY{
		cosY, 0.0, sinY,
		0.0, 1.0, 0.0,
		-sinY, 0.0, cosY
	};
	mat3 rotateZ{
		cosZ, -sinZ, 0.0,
		sinZ, cosZ, 0.0,
		0.0, 0.0, 1.0
	};
	mat3 rotateMat = rotateX * rotateY * rotateZ;
	return Rotate(rotateMat);
}

engOBJECT& engOBJECT::Rotate(const mat3 rotMat)
{
	for (auto& vert : vertices)
	{
		vert = rotMat * vert;
	}

	for (auto& normal : normals)
	{
		normal = rotMat * normal;
	}

	return *this;
}

engOBJECT& engOBJECT::Scale(const vec3& scaleCoef)
{
	for (auto& vert : vertices)
	{
		vert *= scaleCoef;
	}
	return *this;
}

// v  0.0000 1.0000 -0.0000
// [v, 0.0000, 1.0000, -0.0000]
static void AddVertex(const std::vector<std::string>& splitted, OUTPUT<engOBJECT> res)
{
	float x = std::stof(splitted[1]);
	float y = std::stof(splitted[2]);
	float z = std::stof(splitted[3]);
	res->vertices.emplace_back(x, y, z);
}

// vt 0.6667 1.0000 20.9520
// [vt, 0.6667, 1.0000, 20.9520]
static void AddTextureCord(const std::vector<std::string>& splitted, OUTPUT<engOBJECT> res)
{
	float u, v, w = 0.0;
	u = std::stof(splitted[1]);
	v = std::stof(splitted[2]);
	if (splitted.size() > 3)
	{
		w = std::stof(splitted[3]);
	}

	res->textureCords.emplace_back(u, v, w);
}

// vn 0.4714 0.3333 -0.8165
// [vn, 0.4714, 0.3333, -0.8165]
static void AddNormal(const std::vector<std::string>& splitted, OUTPUT<engOBJECT> res)
{
	float x = std::stof(splitted[1]);
	float y = std::stof(splitted[2]);
	float z = std::stof(splitted[3]);

	res->normals.emplace_back(x, y, z);
}

// f 28/5/18 26/6/18 14/6/18
// [f, 28/5/18, 26/6/18, 14/6/18]

#define GET_NUMBERS(num)								\
	numbers = Split(splitted[num], numbersDelimiter);	\
	int v##num = std::stoi(numbers[0]) - 1;				\
	int t##num = std::stoi(numbers[1]) - 1;				\
	int n##num = std::stoi(numbers[2]) - 1

static void AddPolygon(const std::vector<std::string>& splitted, OUTPUT<engOBJECT> res)
{
	static constexpr const char numbersDelimiter = '/';
	std::vector<std::string> numbers;
	GET_NUMBERS(1);
	GET_NUMBERS(2);
	GET_NUMBERS(3);

	res->polygons.emplace_back(v1, v2, v3, n1, n2, n3, t1, t2, t3);
}

#undef GET_NUMBERS

engOBJECT engOBJECT::ImportFromObj(const std::string& path)
{
	engOBJECT res;
	std::ifstream ifs(path);
	ASSERT_MSG(ifs, "Can not open file %s", path.c_str());
	std::string line;
	while (std::getline(ifs, line))
	{
		if (line.empty())
		{
			continue;
		}
		std::vector<std::string> splitted = Split(line);
		auto type = splitted[0];
		if (type == "v") // vertex
		{
			AddVertex(splitted, &res);
		}
		else if (type == "vt") // texture coord
		{
			AddTextureCord(splitted, &res);
		}
		else if (type == "vn") // normal
		{
			AddNormal(splitted, &res);
		}
		else if (type == "f")
		{
			AddPolygon(splitted, &res);
		}
	}
	return res;
}

std::vector<std::shared_ptr<engOBJECT>> engOBJECT::GenerateLights(float r, float a, float margin, float offset,
	int lightsNum, float lightRadius)
{
	std::vector<vec3> endPoints;

	vec3 center(0.0f);
	int verticesNum = vertices.size();
	for (int i = 0; i < verticesNum; ++i)
		center += vertices[i];

	center = (1.0f / verticesNum) * center;
	float eps = 0.01;

	for (int i = 0; i < verticesNum; ++i) {
		if (ApproxEqual((center - vertices[i]).Length(), r), eps) {
			endPoints.push_back(vertices[i]);
		}
	}

	std::vector<std::pair<int, int>> edges;
	for (int i = 0; i < endPoints.size(); ++i) {
		for (int j = 0; j < endPoints.size(); ++j) {
			if (ApproxEqual((endPoints[i] - endPoints[j]).Length(), a, eps)) {
				bool used = false;
				for (auto& edge : edges)
				{
					if (edge.first == j && edge.second == i) {
						used = true;
						break;
					}
				}
				if (!used) {
					edges.emplace_back(i, j);
				}
			}
		}
	}

	float lineLength = a - 2 * margin;
	float d = lineLength / (lightsNum - 1);

	objCIRCLE dsc(lightRadius);
	std::vector<std::shared_ptr<engOBJECT>> res;

	for (size_t i = 0; i < edges.size(); ++i) {
		int n1 = edges[i].first, n2 = edges[i].second;
		vec3 e1 = endPoints[n1], e2 = endPoints[n2];

		vec3 normal = (0.5f * (center - e1 + (center - e2))).Normalized();

		vec3 dir = e2 - e1;
		vec3 dirNormalized = dir.Normalized();
		vec3 start_point = e1 + dirNormalized * margin;
		for (int j = 0; j < lightsNum; ++j) {
			vec3 pos = start_point + d * j * dirNormalized + normal * offset;

			auto tmp = dsc;

			// rotate to align normal vector of light with normal
			mat3 align = AlignMat(vec3(0.0f, 1.0f, 0.0f), normal);
			// translate light to (pos position + normal * offset)
			tmp.Rotate(align);
			tmp.Move(pos);

			std::shared_ptr<engOBJECT> ptr = std::make_shared<objCIRCLE>(tmp);
			ptr->SetName(name + "_light");
			res.emplace_back(ptr);
		}
	}

	return res;
}