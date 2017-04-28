#include <iostream>
#include <vector>
#include <unistd.h>
#include <bitset>
#include <fstream>
#include "mnist_extraction.hpp"

#define GetCurrentDir getcwd

using namespace std;


/***
 *  Reverse an integer bit-wisely
 */
int reverse_int(int i) {
    unsigned char ch1, ch2, ch3, ch4;
    ch1=i&255;
    ch2=(i>>8)&255;
    ch3=(i>>16)&255;
    ch4=(i>>24)&255;
    return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}


/***
 *  Extract data from *-images-idx3-ubyte files (found at yann.lecun.com/exdb/mnist/)
 * 
 */
void read_mnist_img(int NumberOfImages, int DataOfAnImage,vector<vector<double>> &arr) {
    arr.resize(NumberOfImages,vector<double>(DataOfAnImage));
    ifstream file("./t10k-images-idx3-ubyte",ios::binary);
    //ifstream file("../../data/t10k-images-idx3-ubyte",ios::binary);
    if (file.is_open())
    {
	cout <<" File found" << endl;	
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;
        file.read((char*)&magic_number,sizeof(magic_number));
        magic_number= reverse_int(magic_number);	
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverse_int(number_of_images);
        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows= reverse_int(n_rows);
        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols= reverse_int(n_cols);
	
	cout << "Loading data..." << endl;
        for(int i=0;i<number_of_images;++i)
        {
	    if ((i+1)%1000 == 0)
	      cout << "Image number : "<< i+1 << endl; 
            for(int r=0;r<n_rows;++r)
            {
                for(int c=0;c<n_cols;++c)
                {
                    unsigned char temp=0;
		    file.read((char*)&temp,sizeof(temp));		      
                    arr[i][(n_rows*r)+c]= (double)temp;
                }
            }
        }
    } else {
      cout << "File not found." << endl;
      char cCurrentPath[FILENAME_MAX];

      if (!GetCurrentDir(cCurrentPath, sizeof(cCurrentPath)))
	  {
	    return;
	  }

      cCurrentPath[sizeof(cCurrentPath) - 1] = '\0'; /* not really required */

      cout << "The current working directory is " << cCurrentPath << endl;
    }
}



/***
 *  Extract labels from *-labels-idx1-ubyte files (found at yann.lecun.com/exdb/mnist/)
 * 
 */
void read_mnist_lbl(vector<double> &arr) {
//     ifstream file ("../../data/t10k-labels-idx1-ubyte",ios::binary);
    ifstream file ("./t10k-labels-idx1-ubyte",ios::binary);
    if (file.is_open())
    {
	cout <<" File found" << endl;	
        int magic_number=0;
        int number_of_images=0;
        file.read((char*)&magic_number,sizeof(magic_number));
        magic_number= reverse_int(magic_number);	
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverse_int(number_of_images);
	
	arr.resize(number_of_images);
	
	cout << "Loading labels..." << endl;
        for(int i=0;i<number_of_images;++i)
        {
	    if ((i+1)%1000 == 0)
	      cout << "Image number : "<< i+1 << endl; 
	    unsigned char temp=0;
	    file.read((char*)&temp,sizeof(temp));		      
	    arr[i]= (double)temp;
        }
    } else {
      cout << "File not found." << endl;
      char cCurrentPath[FILENAME_MAX];

      if (!GetCurrentDir(cCurrentPath, sizeof(cCurrentPath)))
	  {
	    return;
	  }

      cCurrentPath[sizeof(cCurrentPath) - 1] = '\0'; /* not really required */

      cout << "The current working directory is " << cCurrentPath << endl;
    }
}


/***
 *  Print pixels values of a mnist image in console
 */
void print_mnist_img(int j, vector<vector<double>> &arr) {
    cout << "Image num : " << j << endl;
    for (int i=0; i < 784; ++i) {
      cout << arr[j][i] << " ";
      if (arr[j][i] < 100)
	cout << " ";
      if (arr[j][i] < 10)
	cout << " ";    
      if ((i+1)%28 ==0)
	cout << endl;
    }    
  }