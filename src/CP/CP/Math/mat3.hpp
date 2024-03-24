template<typename T>
std::ostream& operator<<(std::ostream& os, const MAT3_T<T>& m)
{
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			os << m.Data[i][j] << ' ';
		}
		os << '\n';
	}
	return os;
}

template<typename T>
__host__ __device__ T MAT3_T<T>::Det() const
{
	return Data[0][0] * Data[1][1] * Data[2][2] + Data[1][0] * Data[0][2] * Data[2][1] + Data[2][0] * Data[0][1] * Data[1][2]
		- Data[0][2] * Data[1][1] * Data[2][0] - Data[0][0] * Data[1][2] * Data[2][1] - Data[0][1] * Data[1][0] * Data[2][2];
}

template<typename T>
__host__ __device__ MAT3_T<T> MAT3_T<T>::Inv() const
{
	float d = Det();

	float m11 = (Data[1][1] * Data[2][2] - Data[2][1] * Data[1][2]) / d;
	float m12 = (Data[2][1] * Data[0][2] - Data[0][1] * Data[2][2]) / d;
	float m13 = (Data[0][1] * Data[1][2] - Data[1][1] * Data[0][2]) / d;

	float m21 = (Data[2][0] * Data[1][2] - Data[1][0] * Data[2][2]) / d;
	float m22 = (Data[0][0] * Data[2][2] - Data[2][0] * Data[0][2]) / d;
	float m23 = (Data[1][0] * Data[0][2] - Data[0][0] * Data[1][2]) / d;

	float m31 = (Data[1][0] * Data[2][1] - Data[2][0] * Data[1][1]) / d;
	float m32 = (Data[2][0] * Data[0][1] - Data[0][0] * Data[2][1]) / d;
	float m33 = (Data[0][0] * Data[1][1] - Data[1][0] * Data[0][1]) / d;

	return MAT3_T(
		m11, m12, m13,
		m21, m22, m23,
		m31, m32, m33
	);
}

template<typename T>
__host__ __device__ MAT3_T<T> operator*(const MAT3_T<T>& a, const MAT3_T<T>& b)
{
	MAT3_T<T> res;
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			float sum = 0.0f;
			for (int k = 0; k < 3; ++k) {
				sum += a.Data[i][k] * b.Data[k][j];
			}
			res.Data[i][j] = sum;
		}
	}
	return res;
}

template<typename T>
__host__ __device__ MAT3_T<T> operator+(const MAT3_T<T>& a, const MAT3_T<T>& b)
{
	MAT3_T<T> res;
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j) {
			res.Data[i][j] = a.Data[i][j] + b.Data[i][j];
		}

	return res;
}

template<typename T>
__host__ __device__ MAT3_T<T> operator*(float a, const MAT3_T<T>& m)
{
	MAT3_T<T> res;
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j) {
			res.Data[i][j] = a * m.Data[i][j];
		}
	return res;
}

template<typename T>
__host__ __device__ MAT3_T<T> operator*(const MAT3_T<T>& m, float a)
{
	return a * m;
}

template<typename T>
__host__ __device__ VEC3_T<T> operator*(const MAT3_T<T>& m, const VEC3_T<T>& v)
{
	VEC3_T<T> res;
	res.x = m.Data[0][0] * v.x + m.Data[0][1] * v.y + m.Data[0][2] * v.z;
	res.y = m.Data[1][0] * v.x + m.Data[1][1] * v.y + m.Data[1][2] * v.z;
	res.z = m.Data[2][0] * v.x + m.Data[2][1] * v.y + m.Data[2][2] * v.z;
	return res;
}