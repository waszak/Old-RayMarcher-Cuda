#ifndef PRIMITIVES_H
#define PRIMITIVES_H
#include"camera.h"

struct Sphere {
	point center;
	float size;
	int materialId;
};

struct Triangle {

  Vector a, b, c;
  Vector N;             // wektor normalny do płaszczyzny trójkšta
  float d;            // równanie płaszczyzny: dotProd(N,(x,y,z)) + d = 0
  int materialId;

  Triangle(
    Vector a,
    Vector b,
    Vector c,
    int materialId
  ) :
    a(a), b(b), c(c), materialId(materialId) {
    N = uCrossProd(b-a,c-a);
    d = -dotProd(N, a);
  }
};

inline  __host__ __device__  float area(Vector v1, Vector v2, Vector v3) {
  float a = length(v2 - v1);
  float b = length(v3 - v1);
  float c = length(v3 - v2);
  float p = (a + b + c) / 2.0f;
  return sqrtf(p*(p-a)*(p-b)*(p-c));
}

inline  __host__ __device__  bool hitTriangle( ray r, Triangle T, float *t1) {
  float t = (-T.d - dotProd(T.N, (struct Vector){r.start})) / dotProd(T.N, r.dir);
  if ( !(t > 0.00001f && t < 100000.f) )
    return false;
  Vector ip = (struct Vector){r.start} + r.dir * t;
  float abc = area(T.a , T.b , T.c);
  float pbc = area( ip  , T.b , T.c);
  float apc = area(T.a ,  ip  , T.c);
  float abp = area(T.a , T.b ,  ip );
  if ( pbc + apc + abp > abc + 0.001f )
    return false;
  if( t < *t1 ){
     *t1 = t;
    return true;
  }
  return false;
}




#endif

