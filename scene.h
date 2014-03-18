#ifndef SCENE_H
#define SCENE_H
#include "cuda.h"
#include "cuda_runtime.h"
#include "mojvector.h"
#include "primitives.h"
#include <math.h>


struct Color{
	float red, green, blue;
};


inline __host__ __device__ Color operator *(float c, const Color v){
	return (struct Color){v.red *c, v.green * c, v.blue * c };
}

inline __host__ __device__ Color operator +(const Color p, const Color q){
	return (struct Color){p.red + q.red, p.green + q.green, p.blue + q.blue}; 
}
inline __host__ __device__ Color operator *(const Color p, const Color q){
	return (struct Color){p.red * q.red, p.green * q.green, p.blue * q.blue}; 
}


struct Material{
	float reflection;
	Color diffuse;
	float specularPower;
	Color specular;
};


struct Light {
	point position;
	Color intensity;
};

class Scene{
protected:
	int Width;
	int Height;
	const int NumberOfMaterials; 
	const int NumberOfSpheres;
	const int NumberOfLights;
	const int NumberOfTriangles;
	Material * materials;
	Sphere * spheres;
	Light * lights;
	Triangle * triangles;
public:
  Scene(int width, int height, int nom, int nos, int nol, int noT)
	:
	Width(width), 
	Height(height), 
	NumberOfMaterials(nom),
	NumberOfSpheres(nos),
	NumberOfLights(nol),
	NumberOfTriangles(noT){
		materials =(Material *) malloc(nom * sizeof(Material));
		spheres = (Sphere *)malloc(nos * sizeof(Sphere));
		lights = (Light *)malloc(nol * sizeof(Light));
		triangles=(Triangle *) malloc( noT * sizeof(Triangle));
	}
	virtual ~Scene(){
		free(materials);
		free(spheres);
		free(lights);
		free(triangles);
	}
	
	int getNumberOfMaterials() const{
		return NumberOfMaterials;
	}
	
	int getNumberOfSpheres() const{
		return NumberOfSpheres;
	}
	
	int getNumberOfLights() const{
		return NumberOfLights;
	}
	int getNumberOfTriangles() const{
		return NumberOfTriangles;
	}
	void setWidth(int x){
		Width = x;
	}
	void setHeight(int x){
		Height = x;
	}
	int getWidth() const{
		return Width;
	}
	
	int getHeight() const{
		return Height;
	}
	Material & getMaterial(int n){
		return materials[n];
	}
	
	Sphere & getSphere(int n){
		return spheres[n];
	}
	
	Light & getLight(int n){
		return lights[n];
	}

	Triangle & getTriangle(int n){
		return triangles[n];	
	}
	
	Material * getMaterialsArray(){
		return materials;	
	}
	
	Sphere * getSpheresArray(){
		return spheres;	
	}

	
	Light * getLightsArray(){
		return lights;	
	}
	
	Triangle * getTrianglesArray(){
		return triangles;	
	}

};

#endif
