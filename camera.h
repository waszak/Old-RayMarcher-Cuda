#ifndef CAMERA_H
#define CAMERA_H

#define fl(x) ((float)(x)) 

struct ray {
	point start;
	Vector dir;
};
inline __host__ __device__ ray rayConstruct( Vector pos, Vector dir){
	return (struct ray){pos.dir,dir};
}


struct camera {
  Vector location;          
  Vector up, front, right;  
  int Width, Height;
 
  camera(Vector location, Vector up, Vector front, unsigned int xRes, unsigned int yRes) :
    location(location),
    up(up),
    front(front),
    right(uCrossProd(front,up)),
    Width(xRes),
    Height(yRes)
    { }
};

// odchylenie poziome od środka obrazu/obiektywu
inline __host__ __device__ float xA(camera c, float x)
{  return (2.0f * x / fl(c.Width) -1.0f);  }
 
// odchylenie pionowe od środka obrazu/obiektywu
inline __host__ __device__ float yA(camera c, float y)
{  return (2.0f * y / fl(c.Height) -1.0f);  }
 
// przekształcenie wektora u na wektor o długości 1
inline __host__ __device__ Vector unitise(Vector u)//ss
{ return VectorConstruct(u.dir.x, u.dir.y, u.dir.z, 1.0f); }

__host__ __device__ inline ray primaryRay(camera c, float x, float y) { 
  return rayConstruct(
    c.location, 
    unitise(c.up * yA(c,y) + c.right * xA(c,x) + c.front)
  );
}
#endif
