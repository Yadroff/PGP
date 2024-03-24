template<typename T>
__host__ __device__ float VEC3_T<T>::Length() const
{
	return sqrt(x * x + y * y + z * z);
}

template<typename T>
__host__ __device__ VEC3_T<T> VEC3_T<T>::Normalized() const
{
	float l = Length();
	return { x / l, y / l, z / l };
}

template<typename T>
__host__ __device__ VEC3_T<T> operator+(const VEC3_T<T>& a, const VEC3_T<T>& b)
{
	return { a.x + b.x, a.y + b.y, a.z + b.z };
}

template<typename T>
__host__ __device__ VEC3_T<T>& VEC3_T<T>::operator+=(const VEC3_T<T>& other)
{
	x += other.x;
	y += other.y;
	z += other.z;
	return *this;
}

template<typename T>
__host__ __device__ VEC3_T<T> VEC3_T<T>::operator-() const
{
	return { -x, -y, -z };
}

template<typename T>
__host__ __device__ VEC3_T<T> operator-(const VEC3_T<T>& a, const VEC3_T<T>& b)
{
	return { a.x - b.x, a.y - b.y, a.z - b.z };
}

template<typename T>
__host__ __device__ VEC3_T<T> operator*(const T& c, const VEC3_T<T>& v)
{
	return { c * v.x, c * v.y, c * v.z };
}

template<typename T>
__host__ __device__ VEC3_T<T> operator*(const VEC3_T<T>& v, T c)
{
	return c * v;
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const VEC3_T<T>& vec)
{
	os << vec.x << ' ' << vec.y << ' ' << vec.z;
	return os;
}

template<typename T>
std::istream& operator>>(std::istream& is, VEC3_T<T>& vec)
{
	is >> vec.x >> vec.y >> vec.z;
	return is;
}

template<typename T>
__host__ __device__ VEC3_T<T>& VEC3_T<T>::operator*=(const VEC3_T<T>& other)
{
	x *= other.x;
	y *= other.y;
	z *= other.z;
	return *this;
}

template<typename T>
__host__ __device__ VEC3_T<T> operator*(const VEC3_T<T>& lhs, const VEC3_T<T>& rhs)
{
	VEC3_T<T> temp(lhs);
	temp *= rhs;
	return temp;
}