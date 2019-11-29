#ifndef result_set_h_
#define result_set_h_

#include <cstdio>
#include <limits>
#include <vector>
#include <iostream>

namespace panene {

    template <typename IDType, typename DistanceType>
    struct Neighbor {
        IDType id;
        DistanceType dist;

        Neighbor() = default;
        Neighbor(IDType id_, DistanceType dist_) : id(id_), dist(dist_) {}

        friend std::ostream& operator<<(std::ostream& os, const Neighbor<IDType, DistanceType>& obj) {
            os << "(" << obj.id << ", " << obj.dist << ")";
            return os;
        }

        bool operator< (const Neighbor &n) const {
            return this->dist < n.dist;
        }

        bool operator> (const Neighbor &n) const {
            return this->dist > n.dist;
        }

        bool operator== (const Neighbor &n) const {
            return this->id == n.id;
        }

        bool operator!= (const Neighbor &n) const {
            return !(this->id == n.id);
        }
    };

    template <typename IDType, typename DistanceType>
    struct ResultSet {
        ResultSet() = default;
        ResultSet(size_t k_) : k(k_) {
            worstDist = (std::numeric_limits<DistanceType>::max)();

            ids.resize(k);
            distances.resize(k);

            for (size_t i = 0; i < k; ++i) {
                ids[i] = -1;
                distances[i] = worstDist;
            }
        }

        const Neighbor<IDType, DistanceType> operator[](size_t index) const {
            return Neighbor<IDType, DistanceType>(ids[index], distances[index]);
        }

        const IDType * getNeighbors() {
            return &ids[0];
        }

        const DistanceType * getDistances() {
            return &distances[0];
        }

        bool full() const
        {
            return worstDist < (std::numeric_limits<DistanceType>::max)();
        }

        friend std::ostream& operator<<(std::ostream& os, const ResultSet<IDType, DistanceType> &obj) {
            for (size_t i = 0; i < obj.k; ++i) {
                os << i << ":" << obj[i] << " ";
            }
            return os;
        }

        void operator<<(const Neighbor<IDType, DistanceType> &neighbor) {
            if (neighbor.dist >= worstDist) return;

            int i;
            for (i = k - 1; i >= 0; --i) {
                if (ids[i] == neighbor.id) return;
                if (distances[i] < neighbor.dist) break;
            }

            // insert neighbor to (i + 1)
            size_t pos = i + 1;

            // shift (i+1) ~ k - 2
            for (size_t i = k - 1; i > pos; --i) {
                ids[i] = ids[i - 1];
                distances[i] = distances[i - 1];
            }

            ids[pos] = neighbor.id;
            distances[pos] = neighbor.dist;

            worstDist = distances[k - 1];
        }

        size_t k;
        DistanceType worstDist;
        std::vector<IDType> ids;
        std::vector<DistanceType> distances;
    };

}

#endif
