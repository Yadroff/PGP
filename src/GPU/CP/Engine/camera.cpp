#include "camera.h"

rendSETTINGS engRenderSettings;

rendSETTINGS& engGetRenderSettings()
{
	return engRenderSettings;
}

void engCAMERA::BuildLookAtMatrix(const vec3& tmp)
{
	vec3 forward = (position - eye).Normalized();
	vec3 right = vec3::CrossProduct(tmp.Normalized(), forward);
	vec3 up = vec3::CrossProduct(forward, right);

	lookat.Data[0][0] = right.x;
	lookat.Data[1][0] = right.y;
	lookat.Data[2][0] = right.z;
	lookat.Data[3][0] = 0.0;

	lookat.Data[0][1] = up.x;
	lookat.Data[1][1] = up.y;
	lookat.Data[2][1] = up.z;
	lookat.Data[3][1] = 0.0;

	lookat.Data[0][2] = forward.x;
	lookat.Data[1][2] = forward.y;
	lookat.Data[2][2] = forward.z;
	lookat.Data[3][2] = 0.0;

	lookat.Data[0][3] = position.x;
	lookat.Data[1][3] = position.y;
	lookat.Data[2][3] = position.z;
	lookat.Data[3][3] = 1.0;
}