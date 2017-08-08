#pragma once
//defines
#define IMAGESQUANTITY  4   //cantidad inicial de imagenes
#define POINTSQUANTITY 100  //cantidad inicial de puntos
#define INF 90000000		//infinito
#define POINT_SIZE 20		//tama~no de punto
#define THICKNESS 5			//grosor de linea
#define MINMATCHES 7        // minimo numero de matches para hallar F
#define PI 3.1415926535897932384626433832795

static const double DEG_TO_RAD = 0.017453292519943295769236907684886;
static const double EARTH_RADIUS_IN_METERS = 6372797.560856;

//enums
enum Graficar { ALL_IN_ONE, ONE_BY_ONE };