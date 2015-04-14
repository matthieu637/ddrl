#include "UNLF2.hpp"

#include "bib/Logger.hpp"
namespace OPTPP {

void init_never_call(int, ColumnVector&)
{
    LOG_ERROR("I should never be called !");
    exit(1);
}

void init_called(int ndim, ColumnVector& x, const std::vector<float>& v)
{
    ASSERT(static_cast<uint>(ndim) == v.size(), "");

    for (uint i = 1; i <= (uint) ndim; i++)
        x(i) = v[i - 1];
}

}