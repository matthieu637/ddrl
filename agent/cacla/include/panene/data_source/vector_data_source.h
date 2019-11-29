#ifndef panene_vector_data_source_h
#define panene_vector_data_source_h

#include <vector>
#include <data_source/data_source.h>

namespace panene
{

    template<typename T, class D>
    class VectorDataSource
    {
        USE_DATA_SOURCE_SYMBOLS

    public:
        VectorDataSource(const std::vector<std::vector<ElementType>> &data_) : distance(Distance()) {
            data = &data_;
            n = data->size();
            d = (*data)[0].size();
        }

        inline ElementType get(const IDType &id, const IDType &dim) const {
            return (*data)[id][dim];
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
            return distance.squared((*data)[id1].begin(), (*data)[id2].begin(), d);
        }

        DistanceType getSquaredDistance(const IDType &id1, const std::vector<ElementType> &vec2) const {
            return distance.squared((*data)[id1].begin(), vec2.begin(), d);
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

    protected:
        std::vector<std::vector<ElementType>> *data;
        Distance distance;
    };

}
#endif
