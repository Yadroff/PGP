template<typename T>
__host__ __device__ float VEC4_T<T>::Length() const
{
	return sqrt(x * x + y * y + z * z + w * w);
}

template<typename T>
__host__ __device__ VEC4_T<T> VEC4_T<T>::Normalized() const
{
	float len = Length();
	return { x / len, y / len, z / len, w / len };
}

template<typename T>
__host__ __device__ VEC4_T<T> operator+(const VEC4_T<T>& a, const VEC4_T<T>& b)
{
	VEC4_T<T> temp(a);
	temp += b;
	return temp;
}

template<typename T>
__host__ __device__ VEC4_T<T>& VEC4_T<T>::operator+=(const VEC4_T<T>& vec)
{
	x += vec.x;
	y += vec.y;
	z += vec.z;
	return *this;
}

template<typename T>
__host__ __device__ VEC4_T<T> VEC4_T<T>::operator-() const
{
	return { -x, -y, -z, -w };
}

template<typename T>
__host__ __device__ VEC4_T<T> operator-(const VEC4_T<T>& a, const VEC4_T<T>& b)
{
	return a + (-b);
}

template<typename T>
__host__ __device__ VEC4_T<T>& VEC4_T<T>::operator-=(const VEC4_T<T>& vec)
{
	return *this += (-vec);
}

template<typename T>
__host__ __device__ VEC4_T<T> operator*(float c, const VEC4_T<T>& v)
{
	return { c * v.x, c * v.y, c * v.z, c * v.w };
}

template<typename T>
__host__ __device__ VEC4_T<T> operator*(const VEC4_T<T>& v, float c)
{
	return c * v;
}

template<typename T>
__host__ __device__ VEC4_T<T>& VEC4_T<T>::operator*=(const VEC4_T<T>& other)
{
	x *= other.x;
	y *= other.y;
	z *= other.z;
	w *= other.w;

	return *this;
}

template<typename T>
__host__ __device__ VEC4_T<T> operator*(const VEC4_T<T>& lhs, const VEC4_T<T>& rhs)
{
	VEC4_T<T> temp(lhs);
	temp *= rhs;
	return temp;
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const VEC4_T<T>& f)
{
	os << f.x << ' ' << f.y << ' ' << f.z << ' ' << f.w;
	return os;
}

template<typename T>
std::istream& operator>>(std::istream& is, VEC4_T<T>& f)
{
	is >> f.x >> f.y >> f.z >> f.w;
	return is;
}

template<typename T>
__host__ __device__ VEC3_T<T> VEC3_T<T>::CrossProduct(const VEC3_T<T>& a, const VEC3_T<T>& b)
{
	VEC3_T<T> res;
	res.x = a.y * b.z - a.z * b.y;
	res.y = a.z * b.x - a.x * b.z;
	res.z = a.x * b.y - a.y * b.x;

	return res;
}

template<typename T>
__host__ __device__ T VEC3_T<T>::DotProduct(const VEC3_T<T>& other) const
{
	return x * other.x + y * other.y + z * other.z;
}