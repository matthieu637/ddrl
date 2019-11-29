#ifndef panene_binary_data_source_h
#define panene_binary_data_source_h

#include <algorithm>
#include <string>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <utility>
#include <sys/stat.h>
#include <data_source/data_source.h>

namespace panene
{

    template<typename T, class D>
    class BinaryDataSource
    {
        USE_DATA_SOURCE_SYMBOLS

    public:

        BinaryDataSource(const std::string& name_ = "data") : name(name_), distance(Distance()) {
        }

        ~BinaryDataSource() {
            if (opened) delete[] data;
        }

        size_t open(const std::string& path, size_t n_, size_t d_) {
            FILE *fd;

#ifdef _WIN32
            auto err = fopen_s(&fd, path.c_str(), "rb");
#else
            fd = fopen(path.c_str(), "rb");
#endif

            if (!fd) {
                std::cerr << "file " << path << " does not exist" << std::endl;
                throw;
            }

            struct stat sb;
            stat(path.c_str(), &sb);
            n = (std::min)((size_t)sb.st_size / (sizeof(float) * d_), n_);
            d = d_;

            data = new ElementType[n * d];
            opened = true;

            auto ret = fread(data, sizeof(ElementType), n * d, fd);
            fclose(fd);
            return n;
        }

        inline ElementType get(const IDType &id, const IDType &dim) const {
            return *(data + id * d + dim);
        }

        void get(const IDType &id, std::vector<ElementType> &result) const {
            for (size_t i = 0; i < d; ++i) {
                result[i] = get(id, i);
            }
        }

        IDType findDimWithMaxSpan(const IDType &id1, const IDType &id2) {
            size_t dim = 0;
            ElementType maxSpan = 0;

            for (size_t i = 0; i < d; ++i) {
                ElementType span = std::abs(this->get(id1, i) - this->get(id2, i));
                if (maxSpan < span) {
                    maxSpan = span;
                    dim = i;
                }
            }

            return dim;
        }

        void computeMeanAndVar(const IDType *ids, int count, std::vector<DistanceType> &mean, std::vector<DistanceType> &var) {
            mean.resize(d);
            var.resize(d);

            for (size_t i = 0; i < d; ++i)
                mean[i] = var[i] = 0;

            for (int j = 0; j < count; ++j) {
                for (size_t i = 0; i < d; ++i) {
                    mean[i] += this->get(ids[j], i);
                }
            }

            DistanceType divFactor = DistanceType(1) / count;

            for (size_t i = 0; i < d; ++i) {
                mean[i] *= divFactor;
            }

            /* Compute variances */
            for (int j = 0; j < count; ++j) {
                for (size_t i = 0; i < d; ++i) {
                    DistanceType dist = this->get(ids[j], i) - mean[i];
                    var[i] += dist * dist;
                }
            }

            for (size_t i = 0; i < d; ++i) {
                var[i] *= divFactor;
            }
        }

        DistanceType getSquaredDistance(const IDType &id1, const IDType &id2) const {
            return distance.squared(data + id1 * d, data + id2 * d, d);
        }

        DistanceType getSquaredDistance(const IDType &id1, const std::vector<ElementType> &vec2) const {
            return distance.squared(data + id1 * d, vec2.begin(), d);
        }

        size_t size() const {
            return n;
        }

        size_t capacity() const {
            return n;
        }

        size_t dim() const {
            return d;
        }

        size_t n;
        size_t d;
        bool opened = false;
        std::string name;

    protected:
        ElementType * data;
        Distance distance;
    };

}
#endif
