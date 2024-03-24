template<typename T>
std::ostream& operator<<(std::ostream& os, const MAT4_T<T>& m)
{
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			os << m.m[i][j] << ' ';
		}
		os << '\n';
	}
	return os;
}

template<typename T>
__host__ __device__ VEC4_T<T> operator*(const MAT4_T<T>& m, const VEC4_T<T>& v)
{
	VEC4_T<T> res;
	res.x = m.Data[0][0] * v.x + m.Data[0][1] * v.y + m.Data[0][2] * v.z + m.Data[0][3] * v.w;
	res.y = m.Data[1][0] * v.x + m.Data[1][1] * v.y + m.Data[1][2] * v.z + m.Data[1][3] * v.w;
	res.z = m.Data[2][0] * v.x + m.Data[2][1] * v.y + m.Data[2][2] * v.z + m.Data[2][3] * v.w;
	res.w = m.Data[3][0] * v.x + m.Data[3][1] * v.y + m.Data[3][2] * v.z + m.Data[3][3] * v.w;
	return res;
}