#ifndef KERNEL_HPP_INCLUDED
#define KERNEL_HPP_INCLUDED

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <bitset>
class Kernel {
public :
    Kernel(int n_kernels_per_dims,int n_dims);
    Kernel();
    void setWeights(const Eigen::VectorXf _weights);
    double getValue(const std::vector<double>& _state, const int& _dim);
    double getWeight(const int& index) const;
    unsigned int getSize(){return n_basis;};
  const double PI = 3.14159265358979f;
// Convert the 32-bit binary encoding into hexadecimal
static int Binary2Hex( std::string Binary )
{
    std::bitset<32> set(Binary);
    int hex = set.to_ulong();

    return hex;
}

// Convert the 32-bit binary into the decimal
static double GetFloat32( std::string Binary )
{
    int HexNumber = Binary2Hex( Binary );

    bool negative  = !!(HexNumber & 0x80000000);
    int  exponent  =   (HexNumber & 0x7f800000) >> 23;
    int sign = negative ? -1 : 1;

    // Subtract 127 from the exponent
    exponent -= 127;

    // Convert the mantissa into decimal using the
    // last 23 bits
    int power = -1;
    double total = 0.0;
    for ( int i = 0; i < 23; i++ )
    {
        int c = Binary[ i + 9 ] - '0';
        total += (double) c * (double) pow( 2.0, power );
        power--;
    }
    total += 1.0;

    double value = sign * (double) pow( 2.0, exponent ) * total;

    return value;
}

// Get 32-bit IEEE 754 format of the decimal value
static std::string GetBinary32( double value )
{
    union
    {
         double input;   // assumes sizeof(double) == sizeof(int)
         int   output;
    }    data;

    data.input = value;

    std::bitset<sizeof(double) * CHAR_BIT>   bits(data.output);

    std::string mystring = bits.to_string<char,
                                          std::char_traits<char>,
                                          std::allocator<char> >();

    return mystring;
}
private :
    Eigen::VectorXi n_basis_per_dim;
    unsigned int n_dims;
    unsigned int n_basis;
    Eigen::MatrixXf centers;
    Eigen::VectorXf widths;
    Eigen::VectorXf weights;

};

#endif // KERNEL_HPP_INCLUDED
