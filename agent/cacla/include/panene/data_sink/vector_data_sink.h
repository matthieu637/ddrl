#ifndef panene_vector_data_sink_h
#define panene_vector_data_sink_h

#include <vector>

namespace panene {

    template <typename IDType, typename DistanceType>
    class VectorDataSink {
    public:
        VectorDataSink(size_t size_, size_t k_) : size(size_), k(k_) {
            neighbors.resize(size_);
            distances.resize(size_);

            for (size_t i = 0; i < size_; ++i) {
                neighbors[i].resize(k);
                distances[i].resize(k);
            }
        }

        /*
        const std::vector<IDType>& getNeighbors(IDType id) const {
          return neighbors[id];
        }

        const std::vector<DistanceType>& getDistances(IDType id) const {
          return distances[id];
        }
        */

        void getNeighbors(const IDType id, std::vector<IDType> &res) const {
            for (size_t i = 0; i < k; ++i) {
                res[i] = neighbors[id][i];
            }
        }

        void getDistances(const IDType id, std::vector<DistanceType> &res) const {
            for (size_t i = 0; i < k; ++i) {
                res[i] = distances[id][i];
            }
        }

        void setNeighbors(IDType id, const IDType * neighbors_, const DistanceType * distances_) {
            // we "copy" the neighbors and distances 
            for (size_t i = 0; i < k; ++i) {
                neighbors[id][i] = neighbors_[i];
                distances[id][i] = distances_[i];
            }
        }

    private:
        size_t size;
        size_t k;

        std::vector<std::vector<IDType>> neighbors;
        std::vector<std::vector<DistanceType>> distances;
    };

};

#endif
