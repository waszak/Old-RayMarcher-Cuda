//nvcc -g -lGL -lGLU -lGLEW -lglut -I$CUDA_SDK/C/common/inc -L$CUDA_SDK/C/lib -lcutil_x86_64 main.cu parseScene.cpp

#include <iostream>
#include <iostream>
#include <GL/glew.h>
#include <GL/glut.h>
#include <cuda_gl_interop.h>
#include <cutil.h>

#include "parseScene.h"
#include "scene.h"
#include "cuda.h"
#include "math_functions.h"
#include "mojvector.h"
#include "camera.h"
#include "primitives.h"


camera cam((struct Vector){0,0,0}, (struct Vector){0,0,0}, (struct Vector){0,0,0}, 0,  0);

int SUPERSAMPLING = 0;
bool GPU = true;
using namespace std;

GLuint   pbo = 0;      // OpenGL PBO id.
uint    *d_output;     // CUDA device pointer to PBO data

dim3 blockSize(8,8); // threads
dim3 gridSize;         // set up in initPixelBuffer

namespace ll{
	Scene * x = NULL;
}
__device__ Sphere * SpheresGpu;
__device__ Material * MaterialsGpu;
__device__ Light * LightsGpu;
__device__ Triangle * TrianglesGpu;

__host__ __device__ bool hitSphere(const ray r, const Sphere s, float *t)  { 
    Vector dist = s.center - r.start; 
    float B = r.dir * dist;
    float D = B*B - dist * dist + s.size * s.size; 
    if (D < 0.0f) 
        return false; 
    float t0 = B - sqrtf(D); 
    float t1 = B + sqrtf(D);
    bool retvalue = false;  
    if ((t0 > 0.1f) && (t0 < *t)) 
    {
        *t = t0;
        retvalue = true; 
    } 
    if ((t1 > 0.1f) && (t1 < *t)) 
    {
        *t = t1; 
        retvalue = true; 
    }
    return retvalue; 
 }


__global__ void initim1(uint * d_output, uint imageW, uint imageH) {
    uint x  = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint y  = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint id = __umul24(y, imageW) + x;  // Good for < 16MPix

    if ( x < imageW && y < imageH ) {
        d_output[id] = id;
    }
}


int iDivUp(int a, int b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

void initPixelBuffer(int width, int height) {
    if (pbo) {      // delete old buffer
        cudaGLUnregisterBufferObject(pbo);
        glDeleteBuffersARB(1, &pbo);
    }
    // create pixel buffer object for display
    glGenBuffersARB(1, &pbo);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, width*height*sizeof(GLubyte)*4, 0, GL_STREAM_DRAW_ARB);
    cudaGLRegisterBufferObject(pbo);

    // calculate new grid size
    gridSize = dim3(iDivUp(width, blockSize.x), iDivUp(height, blockSize.y));

    // from display:
    cudaGLMapBufferObject((void**)&d_output, pbo  );
    initim1<<<gridSize, blockSize>>>(d_output, width, height);
    CUT_CHECK_ERROR("Kernel error");
    cudaGLUnmapBufferObject(pbo);
}

template <int N>
struct Supersampling 
{
	static const float step = (N>=3?0.25f:(N==2?0.5f:1.0f));
	static const float ratio = (N>=3?0.0625f:(N==2?0.25f:1.0f));
};



 /*
float XX =0.0f,  YY=0.0f, ZZ = 0.0f;
 
void addRay(Scene *sc, float x, float y, Color * color){
	*color =(Color) {0.0f, 0.0f, 0.0f};
	float coef = 1.0f;
	int it = 0;
	ray viewRay;
	if(true)  viewRay= (struct ray){ {x, y, -1000.0f}, { XX, YY, ZZ +1.0f}};
	else{
		//TODO
	}
	do{
		int currentSphere= -1;
		point newStart;
		{
			float t = 2000.0f;
		

			for (int i = 0; i < sc->getNumberOfSpheres(); i++) { 
				if (hitSphere(viewRay, sc->getSphere(i), &t)) {
					currentSphere = i;
				}
			}

			if (currentSphere == -1)
				break;

			newStart = viewRay.start + t* viewRay.dir;
		}

		Vector n = newStart - sc->getSphere(currentSphere).center;
		{
			float temp = n * n;
			if (temp == 0.0f){ 
				break;
			}
			temp = 1.0f / sqrtf(temp); 
			n = temp * n; 
		}
		Material currentMat = sc->getMaterial(sc->getSphere(currentSphere).materialId);
		
		
		for ( int j = 0; j < sc->getNumberOfLights(); ++j) {
			Light current = sc->getLight(j);
			Vector dist = current.position - newStart;
			float fLightProjection = n *dist;
			if (fLightProjection <= 0.0f){
				continue;
			}
			float t = sqrtf(dist * dist);
			if ( t <= 0.0f ){
				continue;
			}
			ray lightRay;
			lightRay.start = newStart;
			lightRay.dir = (1/t) * dist;
			fLightProjection *= (1/t);
			
			// computation of the shadows
			bool inShadow = false; 
			for ( int i = 0; i < sc -> getNumberOfSpheres(); ++i) {
				if (hitSphere(lightRay, sc->getSphere(i), &t)) {
					inShadow = true;
					break;
				}
			}
			if (!inShadow) {
				// lambert
				float lambert = (lightRay.dir * n) * coef;
				color->red += lambert * current.intensity.red * currentMat.diffuse.red;
				color->green += lambert * current.intensity.green * currentMat.diffuse.green;
				color->blue += lambert * current.intensity.blue * currentMat.diffuse.blue;
			
				// blin
				float fViewProjection = viewRay.dir * n;
				Vector blinnDir = lightRay.dir - viewRay.dir;
				float temp = blinnDir * blinnDir;
				if (temp != 0.0f ){
					float blinn =  (1.0f / sqrtf(temp)) * max(fLightProjection - fViewProjection , 0.0f);
					blinn = coef * powf(blinn, currentMat.specularPower);
					*color = *color + blinn * currentMat.specular  * current.intensity;
				}
			}
		}
		
		coef *= currentMat.reflection;
		float reflet = 2.0f * (viewRay.dir * n);
		viewRay.start = newStart;
		viewRay.dir = viewRay.dir - reflet * n;
		it++;
	}while((coef > 0.0f) && (it < 10));
}	
*/
 __host__ __device__  void addRayGpu(float x, float y, camera cam, Color * color, Sphere * sphereArray, int sphereArraySize,
						Triangle * triangleArray, int trianglArraySize, Material * materialArray, int materialArraySize, Light * lightArray, int lightArraySize){
 
	float coef = 1.0f;
	int it = 0;
	ray viewRay; 
	if(false)  viewRay= (struct ray){ {x, y, -1000.0f}, { 0, 0, 1.0f}};
	else{
		viewRay = primaryRay(cam,x,y);
	}
	int currentSphere;
	//int currentTriangle;

	do{
		currentSphere=-1;
	//	currentTriangle=-1;
		float t = 20000.0f;
		bool Break = false;
		point newStart;
		
		for (int i = 0; i < sphereArraySize; i++) { 

			if (hitSphere(viewRay, sphereArray[i], &t)) {
				currentSphere = i;
			}
		}
		/*
		 * TODO
		for( int i = 0; i < trianglArraySize; i++){
		
			if (hitTriangle(viewRay, triangleArray[i], &t)) {
				currentSphere = -1;
				currentTriangle = i;
			}
		}*/
		if (currentSphere != -1)
		{
			newStart = viewRay.start + t* viewRay.dir;
			
		
		
			Vector n = newStart - sphereArray[currentSphere].center;
			{
				float temp = n * n;
				if (temp == 0.0f){ 
					Break = true;
				}
				if( !Break){
					temp = 1.0f / sqrtf(temp); 
					n = temp * n;
				}
			}
			if( !Break){	
				Material currentMat = materialArray[sphereArray[currentSphere].materialId];
				for ( int j = 0; j <lightArraySize; ++j) {
					Light current = lightArray[j];
					Vector dist = current.position - newStart;
					float fLightProjection = n *dist;
					if (fLightProjection <= 0.0f){
						
					}else{
						float t = sqrtf(dist * dist);
						if ( t > 0.0f ){
							ray lightRay;
							lightRay.start = newStart;
							lightRay.dir = (1/t) * dist;
							fLightProjection *= (1/t);
							
							// computation of the shadows
							bool inShadow = false; 
							for ( int i = 0; !inShadow && i < sphereArraySize; ++i) {
								if (hitSphere(lightRay, sphereArray[currentSphere], &t)) {
									inShadow = true;
								}
							}
							if (!inShadow) {
								// lambert
								float lambert = (lightRay.dir * n) * coef;
								color->red += (lambert * current.intensity.red * currentMat.diffuse.red);
								color->green += (lambert * current.intensity.green * currentMat.diffuse.green);
								color->blue += (lambert * current.intensity.blue * currentMat.diffuse.blue);
								// blin
								float fViewProjection = viewRay.dir * n;
								Vector blinnDir = lightRay.dir - viewRay.dir;
								float temp = blinnDir * blinnDir;
								if (temp != 0.0f ){
									float blinn =  (1.0f / sqrtf(temp)) * max(fLightProjection - fViewProjection , 0.0f);
									blinn = coef * powf(blinn, currentMat.specularPower);
									*color = *color + blinn * currentMat.specular  * current.intensity;
								}
							
							}
						}
					}
				}
				
				coef *= currentMat.reflection;
				float reflet = 2.0f * (viewRay.dir * n);
				viewRay.start = newStart;
				viewRay.dir = viewRay.dir - reflet * n;
				
			}

			it++;
		}
		else{
			it = 10;
		}
		if( Break){
			it = 10;
		}
	}while((coef > 0.0f) && (it < 10));
	 
}

template <int SAMPL>
void drawSceneCPU( Scene * & sc, char *filePath, float X, float Y){


	int width = sc->getWidth();
	int height = sc->getHeight();

	float stepY = sc->getHeight()/Y;
	float stepX = sc->getWidth()/X;

	float cY = 0.0f;
	for (float y = 0; y < sc->getHeight(); ) { 
		
		float cX = 0.0f;
		for (float x = 0; x < sc->getWidth();) {
			Color out = {0, 0 ,0};
				
			//Supersampling loop 
			for (float tmpx = x ; tmpx < x + 1.0f; tmpx += (Supersampling<SAMPL>::step ) ){
			    for (float tmpy = y ; tmpy < y + 1.0f; tmpy += (Supersampling<SAMPL>::step)){
					Color tmp = {0,0,0};
					float antyRatio=Supersampling<SAMPL>::ratio;
					addRayGpu(tmpx, tmpy,cam, &tmp, ll::x->getSpheresArray(),ll::x->getNumberOfSpheres(),
							  ll::x->getTrianglesArray(), ll::x->getNumberOfTriangles(), ll::x->getMaterialsArray() , ll::x->getNumberOfMaterials(),ll::x->getLightsArray() , ll::x->getNumberOfLights());
					out = out +  antyRatio * tmp;
				}
			}
			
			glColor3f( min(out.blue,1.0f),
					 min(out.green, 1.0f),
				min(out.red, 1.0f));
			glVertex3f(x, height- y, 0);
			
			float Tmp = stepX - cX;
			float t =  x + Tmp ;
			cX = (t- x) - Tmp;
			x = t;
		}
		float Tmp = stepY - cY;
		float t =  y + Tmp ;
		cY = (t- y) - Tmp;
		y = t;
	}

}


template <int SAMPL>
__global__ void drawSceneGPU(unsigned int * d_output , camera cam, int sceneWidth, int sceneHeight, float Width, float Height, Sphere * sphereArray, int sphereArraySize, 
							 Triangle * triangleArray, int trianglArraySize, Material * materialArray, int materialArraySize, Light * lightArray, int lightArraySize){
	
	float  stepHeight = sceneHeight / Height; 
	float  stepWidth =  sceneWidth / Width ;
	

	unsigned int x  = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	unsigned int y  = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
	unsigned int id = __umul24(y, (int)Width) + x;  
		
	float xp =  x*stepWidth;
	float yp =  y*stepHeight;
	
	Color tmp = {0,0,0};
	Color out = {0, 0, 0};
	for (float tmpx = xp ; tmpx < xp + 1.0f; tmpx += (Supersampling<SAMPL>::step ) ){
		for (float tmpy = yp ; tmpy < yp + 1.0f; tmpy += (Supersampling<SAMPL>::step)){
			tmp = (Color){0,0,0};
			addRayGpu(tmpx, tmpy, cam, &tmp, sphereArray, sphereArraySize,
					  triangleArray, trianglArraySize, materialArray, materialArraySize, lightArray, lightArraySize);
			out = out +  Supersampling<SAMPL>::ratio * tmp;
		}
	}
	int blue = min(out.blue,1.0f)*255;
	int green = min(out.green, 1.0f)*255;
	int red = min(out.red, 1.0f)*255;
	
	{
		if ( x < Width && y < Height ) {
			if(xp < sceneWidth && yp < sceneHeight) d_output[id] =  red + (green<<8) + (blue<<16);
		}
	}
	
}

void keyboard(unsigned char k, int , int ) {
    if (k==27 || k=='q' || k=='Q') exit(1);
    if (k=='E' || k == 'e') SUPERSAMPLING = (SUPERSAMPLING +1)%3;
	  switch(k) {
		  case 's': case 'S': cam.location= cam.location -cam.up * 6; break;
		  case 'w': case 'W' :cam.location = cam.location +cam.up * 6; break;
		  case 'd': case 'D': cam.location= cam.location + cam.right * 6; break;
		  case 'a': case'A':  cam.location =cam.location - cam.right * 6; break;
		  case 'p': case 'P': cam.location = cam.location + cam.front * 6; break;
		  case 'l': case 'L':cam.location= cam.location - cam.front * 6; break;
	  }

	glutPostRedisplay();
}


int X =640, Y=480;
void reshape(int width, int height) {
	X = width;
	Y = height;
	if(!GPU){
		glViewport(0, 0, (GLsizei)width, (GLsizei)height); 
		glMatrixMode(GL_PROJECTION); 
		glLoadIdentity(); //
		gluPerspective(60, (GLfloat)width / (GLfloat)height, 1.0, 100.0);  
		glMatrixMode(GL_MODELVIEW);
	}else{
		initPixelBuffer(X, Y);
		glViewport(0, 0, X, Y);
		glLoadIdentity();
		glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
	}
	
	
}
GLuint mTextureID;

void displayCPU() {
	char msg[100];
	glMatrixMode (GL_PROJECTION);
	glLoadIdentity ();
	glOrtho (0, ll::x->getWidth(), ll::x->getHeight(), 0, 0, 1);
	glMatrixMode (GL_MODELVIEW);
	
	glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);

	cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventRecord(start,0);
	
	
	glBegin( GL_POINTS  );
		if(SUPERSAMPLING == 0)
			drawSceneCPU<1>(ll::x,NULL, X, Y);
		if(SUPERSAMPLING == 1)
			drawSceneCPU<2>(ll::x,NULL, X, Y);
		if(SUPERSAMPLING == 2)
			drawSceneCPU<3>(ll::x,NULL, X, Y);
	glEnd();
	
	cudaEventCreate(&stop);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start,stop);
    sprintf(msg, "CPU[AA%d] RAY: Time: %0.3f [ms]", SUPERSAMPLING+1, elapsedTime);
	glutSetWindowTitle(msg);
	glutSwapBuffers();
    glutReportErrors();
}

void displayGPU(){
	

	char msg[100];
	
	cudaEvent_t start, stop;
    float elapsedTime;
	
	cudaEventCreate(&start);
    cudaEventRecord(start,0);
	{
		cudaGLMapBufferObject((void**)&d_output, pbo  );
		
		if(SUPERSAMPLING == 0)
			drawSceneGPU<1><<<gridSize, blockSize>>>(d_output, cam, ll::x->getWidth(), ll::x->getHeight(), X, Y, SpheresGpu,ll::x->getNumberOfSpheres(), 
								TrianglesGpu, ll::x->getNumberOfTriangles(), MaterialsGpu , ll::x->getNumberOfMaterials(), LightsGpu , ll::x->getNumberOfLights());
		if(SUPERSAMPLING == 1)
			drawSceneGPU<2><<<gridSize, blockSize>>>(d_output, cam,ll::x->getWidth(), ll::x->getHeight(), X, Y, SpheresGpu,ll::x->getNumberOfSpheres(), 
								TrianglesGpu, ll::x->getNumberOfTriangles(), MaterialsGpu , ll::x->getNumberOfMaterials(), LightsGpu , ll::x->getNumberOfLights());
		if(SUPERSAMPLING == 2)
			drawSceneGPU<3><<<gridSize, blockSize>>>(d_output, cam,ll::x->getWidth(), ll::x->getHeight(), X, Y, SpheresGpu,ll::x->getNumberOfSpheres(),
								TrianglesGpu, ll::x->getNumberOfTriangles(), MaterialsGpu , ll::x->getNumberOfMaterials(), LightsGpu , ll::x->getNumberOfLights());
		
		CUT_CHECK_ERROR("Kernel error");
		cudaGLUnmapBufferObject(pbo );
		cudaThreadSynchronize();
	}
	cudaEventCreate(&stop);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start,stop);
    sprintf(msg, "GPU[AA%d] RAY: Kernel time: %0.3f [ms]",SUPERSAMPLING+1, elapsedTime);
	
	glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);
    glRasterPos2i(0, 0);
    glDrawPixels(X, Y, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	
	glutSetWindowTitle(msg);
    glutSwapBuffers();
    glutReportErrors();
}

void allocGPU(){
	cudaMalloc((void**)&SpheresGpu,  ll::x->getNumberOfSpheres()*sizeof(Sphere));
	cudaMemcpy (SpheresGpu, ll::x->getSpheresArray(), ll::x->getNumberOfSpheres()*sizeof(Sphere), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&MaterialsGpu,  ll::x->getNumberOfMaterials()*sizeof(Material));
	cudaMemcpy (MaterialsGpu, ll::x->getMaterialsArray(), ll::x->getNumberOfMaterials()*sizeof(Material), cudaMemcpyHostToDevice);
	
	cudaMalloc((void**)&LightsGpu,  ll::x->getNumberOfLights()*sizeof(Light));
	cudaMemcpy (LightsGpu, ll::x->getLightsArray(), ll::x->getNumberOfLights()*sizeof(Light), cudaMemcpyHostToDevice);
	
	cudaMalloc((void**)&TrianglesGpu,  ll::x->getNumberOfTriangles()*sizeof(Triangle));
	cudaMemcpy (TrianglesGpu, ll::x->getTrianglesArray(), ll::x->getNumberOfTriangles()*sizeof(Triangle), cudaMemcpyHostToDevice);
}
void cleanup() {
    cudaGLUnregisterBufferObject(pbo);
    glDeleteBuffersARB(1, &pbo);
	delete ll::x;
	cudaFree(SpheresGpu);
	cudaFree(LightsGpu);
	cudaFree(MaterialsGpu);
	cudaFree(TrianglesGpu);
}

int main(int argc, char ** argv){
    //char * filePath = "simple.ray";
	char filePath[100];
	char mode[100];
	if( argc == 2 || argc == 3){
        sscanf(argv[1],"%s",filePath);
        if( argc == 3)
		{
			sscanf(argv[2],"%s",mode);
			if( string(mode) == "-cpu"){
				GPU = false;
			}
		}
    }
    else{
      printf("%s plikSceny.ray (-cpu)\n",argv[0]);
      exit(1);
    }
	
	
	try{
		readFile(ll::x, filePath);
	}catch( exception & e){
		cerr<<e.what()<<endl;
		return 1;
	}

	allocGPU();
	
	camera nowa(
    VectorConstruct(ll::x->getWidth()/2, ll::x->getHeight()/2, -500.f),        // położenie
    VectorConstruct(0.0f, 1000.0f, 0.0f, 1.0f),  // kierunek do góry od położenia
    VectorConstruct(0.0f, 0.0f, 1000.0f, 1.0f),ll::x->getWidth(),ll::x->getHeight());
	cam = nowa;
	
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE );
	glutInitWindowSize(ll::x->getWidth(), ll::x->getHeight());
	glutCreateWindow("Ray Marching");
	if(GPU)glutDisplayFunc(displayGPU);
	else glutDisplayFunc(displayCPU);
	
	glutReshapeFunc(reshape);
	glutKeyboardFunc(keyboard);
	glewInit();
	if (!glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object")) {
		fprintf(stderr, "OpenGL requirements not fulfilled !!!\n");
		exit(-1);
	}
	
	
	X = ll::x->getWidth();
	Y = ll::x->getHeight();
	initPixelBuffer(ll::x->getWidth(), ll::x->getHeight());
	atexit(cleanup);
	glutMainLoop();
	
	return 0;
}