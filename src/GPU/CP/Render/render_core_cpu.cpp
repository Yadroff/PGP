#include <thread>
#include <future>

#include "render_core_cpu.h"
#include "Math/math.cuh"

void rendRenderAsyncCPU(const rendCPU_OPTIONS& options)
{
	std::vector<std::future<void>> futures;
	int xCount = std::thread::hardware_concurrency();
	int yCount = std::thread::hardware_concurrency();
	for (size_t i = 0; i < xCount; ++i)
	{
		for (size_t j = 0; j < yCount; ++j)
		{
			futures.push_back(std::async(std::launch::async, rendRenderCoreCPU, options, i, j, xCount, yCount));
		}
	}
	for (auto& future : futures)
	{
		future.get();
	}
}

void rendRenderCoreCPU(const rendCPU_OPTIONS& opts, int idX, int idY, int offsetX, int offsetY)
{
	for (int i = idY; i < opts.height; i += offsetY) {
		float pixelY = (1.0f - 2.0f * ((i + 0.5f) / opts.height)) * opts.tanFOV / opts.aspectRatio;
		for (int j = idX; j < opts.width; j += offsetX) {
			float pixelX = (2.0f * ((j + 0.5f) / opts.width) - 1.0f) * opts.tanFOV;

			vec3 pixelWorldPos = HomogeneousMult(opts.lookAt, vec3(pixelX, pixelY, -1.0));
			vec3 cameraWorldPos = HomogeneousMult(opts.lookAt, vec3(0.0f));

			int rayCount = 0;

			vec3 rayDir = (pixelWorldPos - cameraWorldPos).Normalized();
			(*opts.frameBuffer)[i * opts.width + j] = rendCastRayCPU(cameraWorldPos, rayDir, opts, 0, AIR_REFRACTIVE_IDX, &rayCount);

			opts.rayCounts[i * opts.width + j] = rayCount + 1;
		}
	}
}

vec3 rendCastRayCPU(const vec3& origin, const vec3& direction, const rendCPU_OPTIONS& opts, int curDepth, float refractive, OUTPUT<int> rayCount)
{
	(*rayCount)++;
	if (curDepth >= opts.maxDepth)
		return opts.backgroundColor;

	int hitObjIdx, hitTriangleIdx;
	float hitU, hitV, hitT;
	vec3 hitPos;
	if (rendHitCPU(origin, direction, opts.objects, &hitObjIdx, &hitPos, &hitTriangleIdx, &hitU, &hitV, &hitT)) {
		const engOBJECT* obj = opts.objects[hitObjIdx].get();
		const engTRIANGLE& trig = obj->polygons[hitTriangleIdx];

		vec3 normal = ((1.0f - hitU - hitV) * obj->normals[trig.n1] + hitU * obj->normals[trig.n2] + hitV * obj->normals[trig.n3]).Normalized();

		vec3 mainColor = rendPhongModelCPU(hitPos, direction, trig, hitU, hitV, *obj, opts.objects, opts.lights);
		float n1 = refractive, n2 = obj->refractive;
		float dp = normal.DotProduct(direction);
		if (dp > 0) { // если луч выходит из среды в воздух
			std::swap(n1, n2);
		}

		vec3 refractedDirection, reflectedDirection;
		if (dp > 0)
		{
			refractedDirection = Refract(direction, -normal, n1, n2);
			reflectedDirection = Reflect(direction, -normal);
		}
		else {
			refractedDirection = Refract(direction, normal, n1, n2);
			reflectedDirection = Reflect(direction, normal);
		}

		vec3 reflectedColor = vec3(0.0f), refractedColor = vec3(0.0f);
		if (!ApproxEqual(obj->transparencyCoeff, 0.0f)) {
			vec3 refractedOrigin = dp < 0 ? hitPos - EPS * normal : hitPos + EPS * normal;
			refractedColor = rendCastRayCPU(refractedOrigin, refractedDirection, opts, curDepth + 1, n2, rayCount);
		}

		if (!ApproxEqual(obj->reflectionCoeff, 0.0f)) {
			vec3 reflected_origin = dp < 0 ? hitPos + EPS * normal : hitPos - EPS * normal;
			reflectedColor = rendCastRayCPU(reflected_origin, reflectedDirection, opts, curDepth + 1, refractive, rayCount);
		}

		return (1.0f - obj->reflectionCoeff - obj->transparencyCoeff) * mainColor + obj->reflectionCoeff * reflectedColor + obj->transparencyCoeff * obj->ambientColor * refractedColor;
	}

	return opts.backgroundColor;
}

bool rendHitCPU(const vec3& origin, const vec3& direction, const std::vector<std::shared_ptr<engOBJECT>>& objects,
	OUTPUT<int> hitObjIdx, OUTPUT<vec3> hitPos, OUTPUT<int> hitTriangleIdx, OUTPUT<float> hitU, OUTPUT<float> hitV,
	OUTPUT<float> hitT)
{
	float minT = std::numeric_limits<float>::max();

	bool hit = false;
	for (int objIdx = 0; objIdx < objects.size(); ++objIdx) {
		engOBJECT* obj = objects[objIdx].get();
		for (int trigIdx = 0; trigIdx < obj->polygons.size(); ++trigIdx) {
			const engTRIANGLE& trig = obj->polygons[trigIdx];

			float t, u, v;
			rendTriangleIntersectionCPU(origin, direction, *obj, trig, &t, &u, &v);

			if (u >= 0.0f && v >= 0.0f && u + v <= 1.0f && t > 0.0f) {
				if (t < minT) {
					minT = t;

					*hitObjIdx = objIdx;
					*hitTriangleIdx = trigIdx;
					*hitPos = origin + t * direction;
					*hitU = u;
					*hitV = v;
				}
				hit = true;
			}
		}
	}

	*hitT = minT;

	return hit;
}

vec3 rendPhongModelCPU(const vec3& hitPos, const vec3& direction, const engTRIANGLE& polygon, float hitU, float hitV,
	const engOBJECT& obj, const std::vector<std::shared_ptr<engOBJECT>>& objects,
	const std::vector<std::shared_ptr<scnLIGHT_SOURCE>>& lights)
{
	vec3 normal = ((1.0f - hitU - hitV) * obj.normals[polygon.n1] + hitU * obj.normals[polygon.n2] + hitV * obj.normals[polygon.n3]).Normalized();

	vec3 ambient(obj.ambientCoeff);
	vec3 diffuse(0.0f);
	vec3 specular(0.0f);

	for (size_t i = 0; i < lights.size(); ++i) {
		vec3 lightPos = lights[i]->pos;
		float hitT = 0.0;
		vec3 L = lightPos - hitPos;
		float d = L.Length();
		L = L.Normalized();

		if (rendShadowRayHitCPU(lightPos, -L, objects, &hitT) && (hitT > d || (hitT > d || (d - hitT < EPS)))) {
			float coef = lights[i]->intensity / (d + DISTANCE_ADDITION);
			diffuse += std::max(obj.diffuseCoeff * coef * L.DotProduct(normal), 0.0f) * lights[i]->color;
			vec3 R = Reflect(-L, normal).Normalized();
			vec3 S = -direction;
			specular += obj.specularCoeff * coef * std::pow(std::max(R.DotProduct(S), 0.0f), obj.shininess) * lights[i]->color;
		}
	}

	if (obj.HasTexture()) {
		vec3 texPos = (1.0f - hitU - hitV) * obj.textureCords[polygon.t1] + hitU * obj.textureCords[polygon.t2] + hitV * obj.textureCords[polygon.t3];
		texPos.x = Clamp(texPos.x);
		texPos.y = Clamp(texPos.y);
		texPos.z = Clamp(texPos.z);
		uchar4 color = obj.ptrTexture->GetColor(texPos);
		return vec3(static_cast<float>(color.x) / 255, static_cast<float>(color.y) / 255, static_cast<float>(color.z) / 255) * (diffuse + 0.2f * ambient) + obj.specularColor * specular;
	}

	return ambient * obj.ambientColor + diffuse * obj.diffuseColor + specular * obj.specularColor;
}

void rendTriangleIntersectionCPU(const vec3& origin, const vec3& dir, const engOBJECT& obj, const engTRIANGLE& trig,
	OUTPUT<float> t, OUTPUT<float> u, OUTPUT<float> v)
{
	vec3 e2 = obj.vertices[trig.v2] - obj.vertices[trig.v1];
	vec3 e3 = obj.vertices[trig.v3] - obj.vertices[trig.v1];

	mat3 mat(
		-dir.x, e2.x, e3.x,
		-dir.y, e2.y, e3.y,
		-dir.z, e2.z, e3.z
	);

	vec3 temp = mat.Inv() * (origin - obj.vertices[trig.v1]);

	*t = temp.x;
	*u = temp.y;
	*v = temp.z;
}

bool rendShadowRayHitCPU(const vec3& origin, const vec3& direction,
	const std::vector<std::shared_ptr<engOBJECT>>& objects, OUTPUT<float> hitT)
{
	float minT = std::numeric_limits<float>::max();
	bool hit = false;
	for (size_t objIdx = 0; objIdx < objects.size(); ++objIdx) {
		const engOBJECT& obj = *objects[objIdx];
		for (size_t trigIdx = 0; trigIdx < obj.polygons.size(); ++trigIdx) {
			const engTRIANGLE& trig = obj.polygons[trigIdx];

			float t, u, v;
			rendTriangleIntersectionCPU(origin, direction, obj, trig, &t, &u, &v);

			if (u >= 0.0f && v >= 0.0f && u + v <= 1.0f && t > 0.0f) {
				if (t < minT) {
					minT = t;
				}
				hit = true;
			}
		}
	}
	*hitT = minT;
	return hit;
}