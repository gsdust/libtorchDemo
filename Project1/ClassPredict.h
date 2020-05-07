#pragma once
#include <torch\script.h>
#include <iostream>
#include <math.h>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cstdlib>


//opencv м╥нд╪Ч
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc.hpp"

int predict(torch::jit::script::Module module, uchar* inputImagepuBuffer);
