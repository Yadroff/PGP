#include "render_core_gpu.cuh"

__device__ void rendTriangleIntersectionGPU(const vec3& origin, const vec3& dir, const engDEV_OBJECT& obj, const engTRIANGLE& trig,
	OUTPUT<float> t, OUTPUT<float> u, OUTPUT<float> v)
{
	vec3 e1 = obj.vertices[trig.v2] - obj.vertices[trig.v1];
	vec3 e2 = obj.vertices[trig.v3] - obj.vertices[trig.v1];

	mat3 m(-dir.x, e1.x, e2.x,
		-dir.y, e1.y, e2.y,
		-dir.z, e1.z, e2.z);

	vec3 temp = m.Inv() * (origin - obj.vertices[trig.v1]);

	*t = temp.x;
	*u = temp.y;
	*v = temp.z;
}

__device__ bool rendShadowRayHitGPU(const vec3& origin, const vec3& dir, engDEV_OBJECT* objects, int objectsNum, OUTPUT<float> hitT)
{
	float tMin = FLT_MAX;
	bool hit = false;

	for (int objIdx = 0; objIdx < objectsNum; ++objIdx) {
		const engDEV_OBJECT& obj = objects[objIdx];
		for (int trigInd = 0; trigInd < obj.trianglesNum; ++trigInd) {
			auto& trig = obj.triangles[trigInd];
			float t, u, v;
			rendTriangleIntersectionGPU(origin, dir, obj, trig, &t, &u, &v);

			if (u >= 0.0f && v >= 0.0f && u + v <= 1.0f && t > 0.0f) {
				if (t < tMin) {
					tMin = t;
				}
				hit = true;
			}
		}
	}

	*hitT = tMin;
	return hit;
}

__device__ bool rendHitGPU(const vec3& origin, const vec3& dir, engDEV_OBJECT* objects, int objectsNum, OUTPUT<int> objHitIdx,
	OUTPUT<vec3> hitPos, OUTPUT<int> trigHitIdx, OUTPUT<float> hitU, OUTPUT<float> hitV, OUTPUT<float> hitT)
{
	float tMin = FLT_MAX;

	bool hit = false;
	for (int objIdx = 0; objIdx < objectsNum; ++objIdx) {
		const engDEV_OBJECT& obj = objects[objIdx];
		for (int trigIdx = 0; trigIdx < obj.trianglesNum; ++trigIdx) {
			const auto& trig = obj.triangles[trigIdx];

			float t, u, v;
			rendTriangleIntersectionGPU(origin, dir, obj, trig, &t, &u, &v);

			if (u >= 0.0f && v >= 0.0f && u + v <= 1.0f && t > 0.0f) {
				if (t < tMin) {
					tMin = t;

					*objHitIdx = objIdx;
					*trigHitIdx = trigIdx;
					*hitPos = origin + t * dir;
					*hitV = v;
					*hitU = u;
				}
				hit = true;
			}
		}
	}

	*hitT = tMin;
	return hit;
}

__device__ vec3 rendCastRayGPU(const vec3& origin, const vec3& direction, const rendGPU_KERNEL_OPTIONS& opts, OUTPUT<int> rayCount)
{
	int stack_top = 0;
	rendGPU_CONTEXT stack[MAX_DEPTH];
	stack[0].origin = origin;
	stack[0].direction = direction;
	stack[0].color = vec3(0.0f);
	stack[0].stage = 0;

	int rays = 0;

	while (stack[0].stage < 3) {
		rendGPU_CONTEXT* top = &stack[stack_top];
		if (stack_top >= opts.maxDepth) {
			// "return" background color
			stack[stack_top - 1].color += top->coef * opts.backgroundColor;
			stack[stack_top - 1].stage++;
			stack_top--;
		}
		else if (top->stage == 3) {
			// return
			stack[stack_top - 1].color += top->coef * top->color;
			stack[stack_top - 1].stage++;
			stack_top--;
		}
		else if (top->stage == 0) {
			int hitObjIdx, hitTrigIdx;
			float hitU, hitV, hitT;
			vec3 hit_pos;
			if (rendHitGPU(top->origin, top->direction, opts.objects, opts.objectsNum, &hitObjIdx, &hit_pos, &hitTrigIdx, &hitU, &hitV, &hitT)) {
				top->objHitIdx = hitObjIdx;
				top->trigHitIdx = hitTrigIdx;
				top->hitPos = hit_pos;
				top->hitT = hitT;
				top->hitU = hitU;
				top->hitV = hitV;

				engDEV_OBJECT& obj = opts.objects[hitObjIdx];

				auto& trig = obj.triangles[hitTrigIdx];

				top->color = (1.0f - obj.reflectionCoeff - obj.transparencyCoeff) * rendPhongModelGPU(hit_pos, top->direction, trig, hitU, hitV, obj, opts.objects, opts.objectsNum, opts.lights, opts.lightsNum);

				vec3 normal = ((1.0f - hitU - hitV) * obj.normals[trig.n1] + hitU * obj.normals[trig.n2] + hitV * obj.normals[trig.n3]).Normalized();

				top->normal = normal;

				float n1 = AIR_REFRACTIVE_IDX, n2 = obj.refractive;
				float dotProduct = normal.DotProduct(top->direction);
				if (dotProduct > 0) { // если луч выходит из среды в воздух
					Swap(&n1, &n2);
				}

				top->n1 = n1;
				top->n2 = n2;

				top->stage++;
			}
			else {
				// "return" background color
				top->color = opts.backgroundColor;
				top->stage = 3;
			}
		}
		else if (top->stage == 1) {
			// place new reflection task
			float dotProduct = top->normal.DotProduct(top->direction);
			vec3 reflectedOrigin, reflectedDirection;
			if (dotProduct > 0) {
				reflectedOrigin = top->hitPos - EPS * top->normal;
				reflectedDirection = Reflect(top->direction, -top->normal);
			}
			else {
				reflectedOrigin = top->hitPos + EPS * top->normal;
				reflectedDirection = Reflect(top->direction, top->normal);
			}

			if (!ApproxEqual(opts.objects[top->objHitIdx].reflectionCoeff, 0.0f)) {
				stack_top++;
				stack[stack_top].stage = 0;
				stack[stack_top].coef = vec3(opts.objects[top->objHitIdx].reflectionCoeff);
				stack[stack_top].origin = reflectedOrigin;
				stack[stack_top].direction = reflectedDirection;
				stack[stack_top].color = vec3(0.0f);
				rays++;
			}
			else {
				top->stage++;
			}
		}
		else if (top->stage == 2) {
			// place new refraction task
			float dotProduct = top->normal.DotProduct(top->direction);

			vec3 refractedOrigin, refractedDirection;
			if (dotProduct > 0) {
				refractedOrigin = top->hitPos + EPS * top->normal;
				refractedDirection = Refract(top->direction, -top->normal, top->n1, top->n2);
			}
			else {
				refractedOrigin = top->hitPos - EPS * top->normal;
				refractedDirection = Refract(top->direction, top->normal, top->n1, top->n2);
			}

			if (!ApproxEqual(opts.objects[top->objHitIdx].transparencyCoeff, 0.0f)) {
				stack_top++;
				stack[stack_top].stage = 0;
				stack[stack_top].coef = vec3(opts.objects[top->objHitIdx].transparencyCoeff) * opts.objects[top->objHitIdx].ambientColor;
				stack[stack_top].origin = refractedOrigin;
				stack[stack_top].direction = refractedDirection;
				stack[stack_top].color = vec3(0.0f);
				rays++;
			}
			else {
				top->stage++;
			}
		}
	}

	*rayCount = rays;
	return stack[0].color;
}

__global__ void rendRenderCoreGPU(rendGPU_KERNEL_OPTIONS opts)
{
	int idX = threadIdx.x + blockIdx.x * blockDim.x;
	int idY = threadIdx.y + blockIdx.y * blockDim.y;

	int offsetX = blockDim.x * gridDim.x;
	int offsetY = blockDim.y * gridDim.y;

	for (int i = idY; i < opts.height; i += offsetY) {
		float pixelY = (1.0f - 2.0f * ((i + 0.5f) / opts.height)) * opts.tanFOV / opts.aspectRatio;
		for (int j = idX; j < opts.width; j += offsetX) {
			float pixelX = (2.0f * ((j + 0.5f) / opts.width) - 1.0f) * opts.tanFOV;

			vec3 pixelWorldPos = HomogeneousMult(opts.lookAt, vec3(pixelX, pixelY, -1.0));
			vec3 cameraWorldPos = HomogeneousMult(opts.lookAt, vec3(0.0f));

			int rayCount;

			vec3 rayDir = (pixelWorldPos - cameraWorldPos).Normalized();
			opts.frameBuffer[i * opts.width + j] = rendCastRayGPU(cameraWorldPos, rayDir, opts, &rayCount);

			opts.rayCounts[i * opts.width + j] = rayCount + 1;
		}
	}
}

__device__ vec3 rendPhongModelGPU(const vec3& pos, const vec3& direction, const engTRIANGLE& trig, float u, float v,
	const engDEV_OBJECT& obj, engDEV_OBJECT* objects, int objNum, scnDEV_LIGHT_SOURCE* lights, int lightsNum)
{
	vec3 normal = ((1.0f - u - v) * obj.normals[trig.n1] + u * obj.normals[trig.n2] + v * obj.normals[trig.n3]).Normalized();

	vec3 ambient(obj.ambientCoeff);
	vec3 diffuse(0.0f);
	vec3 specular(0.0f);

	for (int i = 0; i < lightsNum; ++i) {
		vec3 light_pos = lights[i].pos;
		float hit_t = 0.0;
		vec3 L = light_pos - pos;
		float d = L.Length();
		L = L.Normalized();

		if (rendShadowRayHitGPU(light_pos, -L, objects, objNum, &hit_t) && (hit_t > d || (d - hit_t < EPS))) {
			float coef = lights[i].intensity / (d + DISTANCE_ADDITION);
			diffuse += max(obj.diffuseCoeff * coef * L.DotProduct(normal), 0.0f) * lights[i].color;
			vec3 R = Reflect(-L, normal).Normalized();
			vec3 S = -direction;
			specular += obj.specularCoeff * coef * pow(max(R.DotProduct(S), 0.0f), obj.shininess) * lights[i].color;
		}
	}

	if (obj.HasTexture()) {
		vec3 tex_pos = (1.0f - u - v) * obj.textureCoords[trig.t1] + u * obj.textureCoords[trig.t2] + v * obj.textureCoords[trig.t3];
		uchar4 color = obj.texture->GetColor(tex_pos);
		return vec3(static_cast<float>(color.x) / 255, static_cast<float>(color.y) / 255, static_cast<float>(color.z) / 255) * (diffuse + 0.2f * ambient) + obj.specularColor * specular;
	}
	return ambient * obj.ambientColor + diffuse * obj.diffuseColor + specular * obj.specularColor;
}