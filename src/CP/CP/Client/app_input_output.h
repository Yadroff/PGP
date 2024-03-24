#pragma once

#include "../Common/common.cuh"
#include "../Math/vec.cuh"

struct appINPUT_SETTINGS
{
public:
	struct LightParams
	{
		vec3 pos;
		vec3 color;
	};

	struct FloorParams
	{
		static constexpr int POINT_NUMBER = 4;
		vec3 points[POINT_NUMBER];
		std::string texturePath;
		vec3 color;
		float reflectionCoeff;
	};

	struct ObjectParams
	{
		vec3 center;
		vec3 color;
		float radius;
		float reflectionCoeff;
		float transparencyCoeff;
		int lightsNum;
		std::string name;
	};

	struct CameraMovement
	{
		float r0;
		float z0;
		float phi0;
		float Ar;
		float Az;
		float wr;
		float wz;
		float wphi;
		float pr;
		float pz;
	};
public:
	friend std::istream& operator>>(std::istream& is, appINPUT_SETTINGS::CameraMovement& movement);
	friend std::istream& operator>>(std::istream& is, appINPUT_SETTINGS::ObjectParams& object);
	friend std::istream& operator>>(std::istream& is, appINPUT_SETTINGS::FloorParams& floor);
	friend std::istream& operator>>(std::istream& is, appINPUT_SETTINGS::LightParams& light);
	friend std::istream& operator>>(std::istream& is, appINPUT_SETTINGS& settings);
public:
	void ApplySettings() const;
public:
	CameraMovement GetPositionMovement() const { return pos; }
	CameraMovement GetEyeMovement() const { return eye; }

	std::vector<LightParams> GetLight() const { return light; }
	FloorParams GetFloor() const { return floor; }
	std::vector<ObjectParams> GetObjects() const { return objects; }
	std::string GetOutputPathFormat() const { return outputPathFormat; }
	int Frames() const { return framesNum; }
	int Width() const { return width; }
	int Height() const { return height; }
private:
	int framesNum = 0;
	std::string outputPathFormat;
	int width;
	int height;
	float fov;
	CameraMovement pos;
	CameraMovement eye;
	static constexpr int OBJECT_NUM = 3;
	std::vector<ObjectParams> objects{ OBJECT_NUM };
	static constexpr const char* objectNames[] = { "tetrahedron", "octahedron", "dodecahedron" };
	FloorParams floor;
	int lightNum;
	std::vector<LightParams> light;
	int maxDepth;
	int rayCount;
};
