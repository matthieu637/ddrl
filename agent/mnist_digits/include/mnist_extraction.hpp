#ifndef MNIST_EXTRACTION_H
#define MNIST_EXTRACTION_H


#include <vector>
 
using namespace std;

/***
 *  Reverse an integer bit-wisely
 */
int reverse_int (int i);

/***
 *  Extract data from *-images-idx3-ubyte files (found at yann.lecun.com/exdb/mnist/)
 * 
 */
void read_mnist_img(int NumberOfImages, int DataOfAnImage,vector<vector<double>> &arr);

/***
 *  Extract labels from *-labels-idx1-ubyte files (found at yann.lecun.com/exdb/mnist/)
 * 
 */
void read_mnist_lbl(vector<double> &arr);


/***
 *  Print pixels values of a mnist image in console
 */
void print_mnist_img(int j, vector<vector<double>> &arr);




#endif