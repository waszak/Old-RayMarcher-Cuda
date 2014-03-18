#ifndef MOJVECTOR_H
#define MOJVECTOR_H
#include "math.h"
struct point{
	float x, y, z;
};
struct Vector{
	point dir;
};

inline __host__ __device__ Vector VectorConstruct(float x, float y, float z){
	return (struct Vector){x,y,z};
}


inline __host__ __device__ Vector VectorConstruct(float x, float y, float z, float len) {

		float dirX= x;
		float dirY= y;
		float dirZ = z;
    		float newLen = len / sqrtf(x*x + y*y + z*z);
   	 	dirX *= newLen;  
		dirY *= newLen;  
		dirZ *= newLen;
		return (struct Vector){dirX, dirY, dirZ};		
}


inline __host__ __device__ Vector operator* (Vector a, float t)
{ return VectorConstruct(a.dir.x*t, a.dir.y*t, a.dir.z*t); }
inline __host__ __device__ Vector operator+ (Vector a, Vector b)
{ return VectorConstruct(a.dir.x+b.dir.x, a.dir.y+b.dir.y, a.dir.z+b.dir.z); }
inline __host__ __device__ Vector operator -(const point p, const point q){
	return (struct Vector){p.x - q.x, p.y - q.y, p.z - q.z}; 
}
inline __host__ __device__ Vector operator -(const Vector p, const Vector q){
	return (p.dir -q.dir); 
}
inline __host__ __device__ point operator +(const point p, const Vector q){
	return (struct point){p.x + q.dir.x, p.y + q.dir.y, p.z + q.dir.z}; 
}
inline __host__ __device__ Vector operator *(float c, const Vector v){
	return (struct Vector){v.dir.x *c, v.dir.y * c, v.dir.z * c };
}
//iloczyn skalarny
inline __host__ __device__ float operator * (const Vector v1, const Vector v2 ) {
	return v1.dir.x * v2.dir.x + v1.dir.y * v2.dir.y + v1.dir.z * v2.dir.z;
} 

inline  __host__ __device__  Vector uCrossProd(Vector v1, Vector v2) {
  return VectorConstruct(
    v1.dir.y*v2.dir.z - v2.dir.y*v1.dir.z,
    v1.dir.z*v2.dir.x - v2.dir.z*v1.dir.x,
    v1.dir.x*v2.dir.y - v2.dir.x*v1.dir.y,
    1.0f
  );
}

inline  __host__ __device__  float dotProd(Vector v1, Vector v2)
{ return v1 * v2; }
 
inline  __host__ __device__ float length(Vector v)
{ return sqrt(v.dir.x*v.dir.x + v.dir.y*v.dir.y + v.dir.z*v.dir.z); }


#endif
