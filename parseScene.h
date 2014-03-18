#ifndef PARSESCENE_H
#define PARSESCENE_H
#include <iostream>
#include <fstream>
#include <string>
#include "scene.h"
namespace{
	const std::string scene = "Scene";
	const std::string material = "Material";
	const std::string sphere = "Sphere";
	const std::string triangle = "Triangle";
	const std::string light = "Light";
	const std::string comment = "//";
	const std::string beginBlock = "{";
	const std::string endBlock = "}"; 
	const std::string Width = "Width";
	const std::string Height = "Height";
	const std::string NumberOfMaterials = "NumberOfMaterials";
	const std::string NumberOfLights = "NumberOfLights";
	const std::string NumberOfSpheres = "NumberOfSpheres";
	const std::string NumberOfTriangles = "NumberOfTriangles";
	const std::string Diffuse = "Diffuse";
	const std::string Reflection = "Reflection"; 
	const std::string MaterialId = "MaterialId";
	const std::string Center = "Center";
	const std::string Size = "Size"; 
	const std::string Position = "Position";
	const std::string Intensity = "Intensity";
	const std::string Specular = "Specular";
	const std::string Power = "Power";
	const std::string A = "A";
	const std::string B = "B";
	const std::string C = "C";

	const char CR = '\r';
	inline std::string trim( const std::string& str, const std::string& whitespace = " \t\n\r"){
		const unsigned int strBegin = str.find_first_not_of(whitespace);
		const unsigned int strEnd = str.find_last_not_of(whitespace);
		const unsigned int strRange = strEnd - strBegin + 1;
		return  (strEnd == strBegin ? "": str.substr(strBegin,strRange));
	}
	void readLight(std::string &buffer, Scene * &sc, std::ifstream & input);
	void readSphere( std::string &buffer, Scene * &sc, std::ifstream & input);
	void readMaterial(std::string &buffer, Scene * &sc, std::ifstream & input);
	void readScene(Scene * &sc, std::ifstream & input);
}
using namespace std;
class parseException: public exception{
private:
	string msg;
public:
	parseException(const char * Msg):msg(Msg){
	}
	const char* what() const throw(){
		return msg.c_str();
	}
	~parseException() throw(){
	}

};

void readFile(Scene * & sc, char filePath []) throw(parseException);

#endif
