#include "parseScene.h"
namespace{
	void readScene(Scene * &sc, ifstream & input){
		int width, height, nol, nos, nom, noT;
		string buffer="";
		input>>buffer;
		if( buffer != beginBlock){
			throw(parseException("Wrong format .ray files. Scene not defined correctly"));
		}
		input>>buffer;
		if( buffer != Width ){
			throw(parseException("Wrong format .ray files. Scene not defined correctly."));
		}
		input>>width;
		
		input>>buffer;
		if( buffer != Height){
			throw(parseException("Wrong format .ray files. Scene not defined correctly."));
		}
		input>>height;
		
		input>>buffer;
		if( buffer != NumberOfMaterials){
			throw(parseException("Wrong format .ray files. Scene not defined correctly."));
		}
		input>>nom;
		
		input>>buffer;
		if( buffer != NumberOfSpheres){
			throw(parseException("Wrong format .ray files. Scene not defined correctly."));
		}
		input>>nos;
		
		input>>buffer;
		if( buffer != NumberOfTriangles){
			throw(parseException("Wrong format .ray files. Scene not defined correctly."));
		}
		input>>noT;

		input>>buffer;
		if( buffer != NumberOfLights){
			throw(parseException("Wrong format .ray files. Scene not defined correctly."));
		}
		input>>nol;
		
		input>>buffer;
		if( buffer != endBlock){
			throw(parseException("Wrong format .ray file. Scene not defined correctly"));
		}
		if( width <1 || height <1 || nos < 0 || nol <0 || nom < 0){
			throw(parseException("Wrong format .ray file. Scene not defined correctly"));
		}
		sc = new Scene(width,height, nom, nos, nol, noT);
	}

	void readMaterial(string &buffer, Scene * &sc, ifstream & input){
		int id;
		float r,g,b;
		float reflection;float power;
		
		char buf[1000];
		sscanf(buffer.c_str(),"%s%d", buf,&id);
		if( id < 0 || id >= sc->getNumberOfMaterials()){
			throw(parseException("Wrong format .ray files. Material not defined correctly"));
		}
		Material & pom = sc->getMaterial(id);
		input>>buffer;
		if( buffer != beginBlock){
			throw(parseException("Wrong format .ray files. Material not defined correctly"));
		}

		input>>buffer;
		if( buffer != Diffuse){
			throw(parseException("Wrong format .ray files. Material not defined correctly."));
		}
		input>>r>>g>>b;
		input>>buffer;
		if( buffer != Reflection){
			throw(parseException("Wrong format .ray files. Material not defined correctly."));
		}
		input>>reflection;
		if( r < 0.0f || r >1.0f
			||g < 0.0f || g >1.0f
			||b < 0.0f || b >1.0f
			||reflection <0.0f || reflection >1.0f){
			throw(parseException("Wrong format .ray files. Material not defined correctly"));
		}
		pom.diffuse.red = r; pom.diffuse.green = g; pom.diffuse.blue = b;
		pom.reflection = reflection;
		
		input>>buffer;
		if( buffer != Specular){
			throw(parseException("Wrong format .ray files. Material not defined correctly"));
		}
		input>>r>>g>>b;
		
		input>>buffer;
		if( buffer != Power){
			throw(parseException("Wrong format .ray files. Material not defined correctly"));
		}
		input>>power;
		
		input>>buffer;
		if( buffer != endBlock){
			throw(parseException("Wrong format .ray files. Material not defined correctly"));
		}
		if( r < 0.0f || r >1.0f
			||g < 0.0f || g >1.0f
			||b < 0.0f || b >1.0f
			||power <0.0f || power >100.0f){
			throw(parseException("Wrong format .ray files. Material not defined correctly"));
		}
		pom.specularPower = power;
		pom.specular.red =r;
		pom.specular.green = g;
		pom.specular.blue = b;

	}

	void readSphere( string &buffer, Scene * &sc, ifstream & input){
		int id;
		float x,y,z, size;
		int materialId;
		
		char buf[1000];
		sscanf(buffer.c_str(),"%s%d", buf,&id);
		if( id < 0 || id >= sc->getNumberOfSpheres()){
			throw(parseException("Wrong format .ray files. Sphere not defined correctly"));
		}
		
		Sphere & pom = sc->getSphere(id);
		input>>buffer;
		if( buffer != beginBlock){
			throw(parseException("Wrong format .ray files. Sphere not defined correctly"));
		}

		input>>buffer;
		if( buffer != Center){
			throw(parseException("Wrong format .ray files. Sphere not defined correctly."));
		}
		input>>x>>y>>z;
		input>>buffer;
		if( buffer != Size){
			throw(parseException("Wrong format .ray files. Sphere not defined correctly."));
		}
		input>>size;
		
		input>>buffer;
		if( buffer != MaterialId){
			throw(parseException("Wrong format .ray files. Sphere not defined correctly."));
		}
		input>>materialId;
		
		input>>buffer;
		if( buffer != endBlock){
			throw(parseException("Wrong format .ray files. Sphere not defined correctly"));
		}
		if( size <0 || materialId <0 || materialId >= sc->getNumberOfMaterials()){
			throw(parseException("Wrong format .ray files. Sphere not defined correctly"));
		}
		pom.center.x = x; pom.center.y = y; pom.center.z = z;
		pom.size = size;
		pom.materialId = materialId;
	}

	void readTriangle(string &buffer, Scene * &sc, ifstream & input){
		int id;
		float xa,ya,za, xb,yb,zb, xc,yc,zc;
		int materialId;
		
		char buf[1000];
		sscanf(buffer.c_str(),"%s%d", buf,&id);
		if( id < 0 || id >= sc->getNumberOfTriangles()){
			throw(parseException("Wrong format .ray files. Triangle not defined correctly"));
		}
		

		input>>buffer;
		if( buffer != beginBlock){
			throw(parseException("Wrong format .ray files. Triangle not defined correctly"));
		}
		
		input>>buffer;
		if( buffer != A){
			throw(parseException("Wrong format .ray files. Triangle not defined correctly"));
		}
		input>>xa>>ya>>za;
		
		input>>buffer;
		if( buffer != B){
			throw(parseException("Wrong format .ray files. Triangle not defined correctly"));
		}
		input>>xb>>yb>>zb;
		
		input>>buffer;
		if( buffer != C){
			throw(parseException("Wrong format .ray files. Triangles not defined correctly"));
		}
		input>>xc>>yc>>zc;
		
		input>>buffer;
		if( buffer != MaterialId){
			throw(parseException("Wrong format .ray files. Triangle not defined correctly."));
		}
		input>>materialId;

		input>>buffer;
		if( buffer != endBlock){
			throw(parseException("Wrong format .ray files. Triangle not defined correctly"));
		}
		if(  materialId <0 || materialId >= sc->getNumberOfMaterials()){
			throw(parseException("Wrong format .ray files. Triangle not defined correctly"));
		}
		
		Triangle & pom = sc->getTriangle(id);
		pom.a = (struct Vector) {xa, ya, za};
		pom.b = (struct Vector) {xb, yb, zb};
		pom.c = (struct Vector) {xc, yc, zc};
		pom.materialId = materialId;

	}

	void readLight(string &buffer, Scene * &sc, ifstream & input){
		int id;
		float x,y,z;
		float r,g,b;
	
		char buf[1000];
		sscanf(buffer.c_str(),"%s%d", buf,&id);
		if( id < 0 || id >= sc->getNumberOfLights()){
			throw(parseException("Wrong format .ray files. Light not defined correctly"));
		}
		Light & pom = sc->getLight(id);
		input>>buffer;
		if( buffer != beginBlock){
			throw(parseException("Wrong format .ray files. Light not defined correctly"));
		}

		input>>buffer;
		if( buffer != Position){
			throw(parseException("Wrong format .ray files. Light not defined correctly."));
		}
		input>>x>>y>>z;
		input>>buffer;
		if( buffer != Intensity){
			throw(parseException("Wrong format .ray files. Light not defined correctly."));
		}
		input>>r>>g>>b;
		input>>buffer;
		if( buffer != endBlock){
			throw(parseException("Wrong format .ray files. Light not defined correctly"));
		}
		if( r < 0.0f || r >1.0f
			||g < 0.0f || g >1.0f
			||b < 0.0f || b >1.0f){
			throw(parseException("Wrong format .ray files. Material not defined correctly"));
		}
		pom.intensity = (Color){r,g,b};
		pom.position.x = x; pom.position.y = y; pom.position.z = z;
	}
}

void readFile(Scene * & sc, char filePath [])throw(parseException){
	ifstream input;
	input.open(filePath);
	string buffer = comment;

	while( buffer.substr(0, comment.size()) == comment || buffer == "") {
		std::getline(input, buffer);
		buffer = trim(buffer);
	}
	buffer = trim(buffer);
	if( buffer == scene){
		readScene(sc,input);
 	}else{
		throw(parseException("Wrong format .ray file. Scene not defined correctly"));
	}
	
	for( int i = 0; i < sc->getNumberOfMaterials(); i++){
		std::getline(input, buffer);
		buffer = trim(buffer);
		while( buffer.substr(0, comment.size()) == comment || buffer == "") {
			std::getline(input, buffer);
			buffer = trim(buffer);
		}
		buffer = trim(buffer);
		if( buffer.substr(0, material.size()) == material ){
			readMaterial(buffer, sc, input);
		}else{
			throw(parseException("Wrong format .ray file. Material not defined correctly"));
		}
	}
	
	for( int i = 0; i < sc->getNumberOfSpheres(); i++){
		std::getline(input, buffer);
		buffer = trim(buffer);
		while( buffer.substr(0, comment.size()) == comment || buffer == "") {
			std::getline(input, buffer);
			buffer = trim(buffer);
		}
		buffer = trim(buffer);
		if( buffer.substr(0, sphere.size()) == sphere ){
			readSphere(buffer, sc, input);
		}else{
			throw(parseException("Wrong format .ray file. Sphere not defined correctly"));
		}
	}
	
	for( int i =0; i < sc->getNumberOfTriangles(); i++){
		std::getline(input, buffer);
		buffer = trim(buffer);
		while( buffer.substr(0, comment.size()) == comment || buffer == "") {
			std::getline(input, buffer);
			buffer = trim(buffer);
		}
		buffer = trim(buffer);
		if( buffer.substr(0, triangle.size()) == triangle ){
			readTriangle(buffer, sc, input);
		}else{
			throw(parseException("Wrong format .ray file. Triangle not defined correctly"));
		}
	}
	for( int i = 0; i < sc->getNumberOfLights(); i++){
		std::getline(input, buffer);
		buffer = trim(buffer);
		while( buffer.substr(0, comment.size()) == comment || buffer == "") {
			std::getline(input, buffer);
			buffer = trim(buffer);
		}
		buffer = trim(buffer);
		if( buffer.substr(0, light.size()) == light ){
			readLight(buffer, sc, input);
		}else{
			throw(parseException("Wrong format .ray file. Light not defined correctly"));
		}
	}
	input.close();
}
