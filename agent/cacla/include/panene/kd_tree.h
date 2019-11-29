#ifndef panene_kd_tree_h
#define panene_kd_tree_h

namespace panene {

    struct InsertionLog
    {
        size_t freq;
        size_t depth;

        InsertionLog() = default;
        InsertionLog(size_t freq_, size_t depth_) : freq(freq_), depth(depth_) { }
    };

    /***
     * a KD tree that records the frequency and depth of each leaf
     */

    template<class NodePtr>
    struct KDTree
    {
        NodePtr root;
        size_t size;
        size_t capacity;
        int freqSum = 0;
        double cost;
        std::vector<InsertionLog> insertionLog;

        KDTree(size_t capacity_) : capacity(capacity_) {
            size = freqSum = 0;
            cost = 0;
            root = nullptr;
            insertionLog.resize(capacity);
        }

        ~KDTree() {
            if (root != nullptr)
                root->~Node();
        }

        float computeCost() {
            float cost = 0;

            for (size_t i = 0; i < capacity; ++i) {
                cost += (float)insertionLog[i].freq / freqSum * insertionLog[i].depth;
            }
            return cost;
        }

        float computeImbalance() {
            float ideal = (float)(log(size) / log(2));
            return computeCost() / ideal;
        }

        size_t computeMaxDepth() {
            size_t maxDepth = 0;
            for (size_t i = 0; i < capacity; ++i) {
                if (maxDepth < insertionLog[i].depth)
                    maxDepth = insertionLog[i].depth;
            }
            return maxDepth;
        }

        void setInsertionLog(const size_t id, const size_t freq, const size_t depth) {
            freqSum = freqSum - insertionLog[id].freq + freq;
            insertionLog[id].freq = freq;
            insertionLog[id].depth = depth;
        }

        float getCachedCost() {
            return cost;
        }

        float getCachedImbalance() {
            float ideal = (float)(log(size) / log(2));
            return cost / ideal;
        }

        float incrementFreqAndDepthByOne(const size_t id) {
            size_t depth = insertionLog[id].depth;
            size_t freq = insertionLog[id].freq;

            cost = ((double)freqSum * cost + depth + freq + 1) / (freqSum + 1);

            freqSum++;
            insertionLog[id].freq++;
            insertionLog[id].depth++;

            return (float)cost;
        }

        float incrementFreqByOne(const size_t id) {
            size_t depth = insertionLog[id].depth;

            cost = ((double)freqSum * cost + depth) / (freqSum + 1);

            freqSum++;
            insertionLog[id].freq++;

            return (float)cost;
        }
    };

}
#endif
