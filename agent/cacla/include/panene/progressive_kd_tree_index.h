#ifndef panene_progressive_kd_tree_index_h
#define panene_progressive_kd_tree_index_h

#include <vector>
#include <algorithm>
#include <random>
#include <cstring>
#include <cstdio>
#include <iostream>
#include <queue>
#include <cassert>
#include <map>

#include <panene/kd_tree_index.h>

#ifdef BENCHMARK
#include <util/timer.h>
#define BENCH(x) x
#else
#define BENCH(x) ((void)0)
#endif

namespace panene
{

    enum UpdateStatus {
        NoUpdate,
        BuildingTree,
        InsertingPoints
    };

    struct TreeWeight {
        float addPointWeight;
        float updateIndexWeight;

        TreeWeight(float addPointWeight_, float updateIndexWeight_) : addPointWeight(addPointWeight_), updateIndexWeight(updateIndexWeight_) {
        }
    };

    struct UpdateResult2 {
        size_t numPointsInserted;

        size_t addPointOps;
        size_t updateIndexOps;

        size_t addPointResult;
        size_t updateIndexResult;

        double addPointElapsed;
        double updateIndexElapsed;

        UpdateResult2() = default;

        UpdateResult2(
            size_t addPointOps_, size_t updateIndexOps_,
            size_t addPointResult_, size_t updateIndexResult_,
            size_t numPointsInserted_,
            double addPointElapsed_, double updateIndexElapsed_) :
            numPointsInserted(numPointsInserted_),
            addPointOps(addPointOps_),
            updateIndexOps(updateIndexOps_),
            addPointResult(addPointResult_),
            updateIndexResult(updateIndexResult_),
            addPointElapsed(addPointElapsed_),
            updateIndexElapsed(updateIndexElapsed_)
        {
        }

        friend std::ostream& operator<<(std::ostream& os, const UpdateResult2& obj) {
            os << "UpdateResult2(addPointOps: " << obj.addPointResult << " / " << obj.addPointOps << ", "
                << "updateIndexOps: " << obj.updateIndexResult << " / " << obj.updateIndexOps << ", numPointsInserted: " << obj.numPointsInserted << ")";
            return os;
        }
    };

    template <typename DataSource>
    class ProgressiveKDTreeIndex : public KDTreeIndex<DataSource>
    {
        USE_KDTREE_INDEX_SYMBOLS

            typedef DataSource DataSourceT;

    public:
        ProgressiveKDTreeIndex(DataSource *dataSource_, IndexParams indexParams_, TreeWeight weight_ = TreeWeight(0.3, 0.7), const float reconstructionWeight_ = .25f) : KDTreeIndex<DataSource>(dataSource_, indexParams_, Distance()), weight(weight_), reconstructionWeight(reconstructionWeight_) {
        }

        size_t addPoints(size_t ops) {
            size_t oldSize = size;
            size += ops;
            if (size > dataSource->size())
                size = dataSource->size();

            if (oldSize == 0) { // for the first time, build the index as we did in the non-progressive version.
                buildIndex();
                return ops;
            }
            else {
                for (size_t i = oldSize; i < size; ++i) {
                    for (size_t j = 0; j < numTrees; ++j) {
                        trees[j]->size++;
                        addPointToTree(trees[j], trees[j]->root, i, 0);
                    }
                }

                if (updateStatus == UpdateStatus::InsertingPoints) {
                    for (size_t i = oldSize; i < size && sizeAtUpdate < size; ++i) {
                        ongoingTree->size++;
                        addPointToTree(ongoingTree, ongoingTree->root, sizeAtUpdate++, 0);
                    }
                }
                return size - oldSize;
            }
        }

        void beginUpdate() {
            updateStatus = UpdateStatus::BuildingTree;
            sizeAtUpdate = size;
            ids.resize(sizeAtUpdate);

            for (size_t i = 0; i < sizeAtUpdate; ++i) ids[i] = int(i);
            std::random_shuffle(ids.begin(), ids.end());

            ongoingTree = new KDTree<NodePtr>(dataSource->capacity());
            ongoingTree->root = new(pool) Node(ongoingTree);
            std::queue<NodeSplit> empty;
            queue = empty;
            queue.push(NodeSplit(ongoingTree->root, &ids[0], sizeAtUpdate, 1));

            ongoingTree->size = sizeAtUpdate;
        }

        size_t update(int ops) {
            int updatedCount = 0;

            while ((ops == -1 || updatedCount < ops) && !queue.empty()) {
                NodeSplit nodeSplit = queue.front();
                queue.pop();

#if PANENEDEBUG
                std::cerr << "updatedCount " << updatedCount << std::endl;
#endif

                NodePtr node = nodeSplit.node;
                IDType *begin = nodeSplit.begin;
                int count = nodeSplit.count;
                int depth = nodeSplit.depth;

#if PANENEDEBUG
                std::cerr << begin << " " << count << std::endl;
#endif

                // At this point, nodeSplit the two children of nodeSplit are nullptr
                if (count == 1) {
                    node->child1 = node->child2 = NULL;    /* Mark as leaf node. */
                    node->id = *begin;    /* Store index of this vec. */ // TODO id of vec
                    ongoingTree->setInsertionLog(node->id, 1, depth);
                }
                else {
                    int idx;
                    int cutfeat;
                    DistanceType cutval;
                    meanSplit(begin, count, idx, cutfeat, cutval);

#if PANENEDEBUG        
                    std::cerr << "cut index: " << idx << " cut count: " << count << std::endl;
#endif

                    node->divfeat = cutfeat;
                    node->divval = cutval;
                    node->child1 = new(pool) Node(ongoingTree);
                    node->child2 = new(pool) Node(ongoingTree);

                    queue.push(NodeSplit(node->child1, begin, idx, depth + 1));
                    queue.push(NodeSplit(node->child2, begin + idx, count - idx, depth + 1));
                }
                updatedCount += 1; // count; // std::min(1, count / 2);
            }

            if (updateStatus == UpdateStatus::BuildingTree && queue.empty()) {
                updateStatus = UpdateStatus::InsertingPoints;
            }

            if (updateStatus == UpdateStatus::InsertingPoints) {
                if (ongoingTree->size < size) {
                    // insert points from sizeAtUpdate to size

                    while (ongoingTree->size < size && (ops == -1 || updatedCount < ops)) {
                        ongoingTree->size++;
                        addPointToTree(ongoingTree, ongoingTree->root, sizeAtUpdate, 0);

                        sizeAtUpdate++;
                        updatedCount++;
                    }
                }

                if (ongoingTree->size >= size) {
                    // finished creating a new tree
                    ongoingTree->cost = ongoingTree->computeCost();

                    size_t victimId = 0;
                    float maxImbalance = trees[0]->computeImbalance();

                    // find the most unbalanced one
                    for (size_t i = 1; i < numTrees; ++i) {
                        float imbalance = trees[i]->computeImbalance();

                        if (maxImbalance < imbalance) {
                            maxImbalance = imbalance;
                            victimId = i;
                        }
                    }

                    // get the victim
                    auto victim = trees[victimId];

                    // replace the victim with the newly created tree
                    delete victim;

                    trees[victimId] = ongoingTree;

                    // reset the sizeAtUpdate
                    sizeAtUpdate = 0;
                    updateStatus = UpdateStatus::NoUpdate;
                }
            }

            return updatedCount;
        }

        UpdateResult2 run(size_t ops) {
            size_t addPointOps = 0, updateIndexOps = 0;
            size_t addPointResult = 0, updateIndexResult = 0;
            double addPointElapsed = 0, updateIndexElapsed = 0;

            if (updateStatus != UpdateStatus::NoUpdate) {
                addPointOps = (size_t)(ops * weight.addPointWeight);
                updateIndexOps = (size_t)(ops * weight.updateIndexWeight);
            }
            else {
                addPointOps = ops;
            }

            BENCH(Timer timer);
            BENCH(timer.begin());

            if (addPointOps > 0) {
                addPointResult = addPoints(addPointOps);
            }

            if (addPointResult == 0) {
                // if we added all points, put all operations to update index
                weight.updateIndexWeight += weight.addPointWeight;
                weight.addPointWeight = 0;
                updateIndexOps = ops;
                addPointOps = 0;
            }
            size_t numPointsInserted = size;

            BENCH(addPointElapsed = timer.end());

            if (updateStatus != NoUpdate) {
                BENCH(timer.begin());
                updateIndexResult = update(updateIndexOps);
                BENCH(updateIndexElapsed = timer.end());
            }

            return UpdateResult2(
                addPointOps, updateIndexOps,
                addPointResult, updateIndexResult,
                numPointsInserted,
                addPointElapsed, updateIndexElapsed);
        }

        void checkBeginUpdate() {
            if (updateStatus == UpdateStatus::NoUpdate) {
                float updateCost = (float)std::log2(size) * size;

                if (queryLoss > updateCost * reconstructionWeight) {
                    beginUpdate();
                    queryLoss = 0;
                }
            }
        }

        void knnSearch(
            const std::vector<ElementType> &vector,
            ResultSet<IDType, DistanceType> &resultSet,
            size_t knn,
            const SearchParams& params)
        {
            float costSum = findNeighbors(vector, resultSet, params);

            size_t ideal = std::log2(size);
            queryLoss += costSum - numTrees * ideal;

            checkBeginUpdate();
        }
        
        void knnSearch(
            const IDType &qid,
            ResultSet<IDType, DistanceType> &resultSet,
            size_t knn,
            const SearchParams& params)
        {
            std::vector<ElementType> vector(dim);
            dataSource->get(qid, vector);

            float costSum = findNeighbors(vector, resultSet, params);

            size_t ideal = std::log2(size);
            queryLoss += costSum - numTrees * ideal;

            checkBeginUpdate();
        }

        // this needs to be improved
        void knnSearch(
            const std::vector<IDType> qids,
            std::vector<ResultSet<IDType, DistanceType>> &resultSets,
            size_t knn,
            const SearchParams& params)
        {
            std::vector<std::vector<ElementType>> vectors(qids.size());

            for (size_t i = 0; i < qids.size(); ++i) {
                vectors[i].resize(dim);
                dataSource->get(qids[i], vectors[i]);
            }

            knnSearch(vectors, resultSets, knn, params);
        }

        void knnSearch(
            const std::vector<std::vector<ElementType>> &vectors,
            std::vector<ResultSet<IDType, DistanceType>> &resultSets,
            size_t knn,
            const SearchParams& params)
        {
            resultSets.resize(vectors.size());

            float costSum = 0;

#pragma omp parallel num_threads(params.cores)
            {
#pragma omp for schedule(static) reduction(+:costSum)
                for (int i = 0; i < (int)vectors.size(); i++) {
                    resultSets[i] = ResultSet<IDType, DistanceType>(knn);
                    costSum += findNeighbors(vectors[i], resultSets[i], params);
                }
            }

            queryLoss += costSum;
            checkBeginUpdate();
        }

        // alias for knnSearch(points) since Cython does not seem to support method overloading
        void knnSearchVec(
            const std::vector<std::vector<ElementType>> &vectors,
            std::vector<ResultSet<IDType, DistanceType>> &resultSets,
            size_t knn,
            const SearchParams& params)
        {
            knnSearch(vectors, resultSets, knn, params);
        }

    protected:

        void buildIndex() {
            std::vector<IDType> ids(size);
            for (size_t i = 0; i < size; ++i) {
                ids[i] = IDType(i);
            }

            for (size_t i = 0; i < numTrees; ++i) {
                std::random_shuffle(ids.begin(), ids.end());
                trees[i]->root = divideTree(trees[i], &ids[0], size, 1);
                trees[i]->size = size;
                trees[i]->cost = trees[i]->computeCost();
            }
        }

        void addPointToTree(KDTree<NodePtr>* tree, NodePtr node, IDType id, int depth) {
            if ((node->child1 == NULL) && (node->child2 == NULL)) {
                // if leaf

                size_t nodeId = node->id;
                size_t divfeat = dataSource->findDimWithMaxSpan(id, nodeId);

                NodePtr left = new(pool) Node(tree);
                left->child1 = left->child2 = NULL;

                NodePtr right = new(pool) Node(tree);
                right->child1 = right->child2 = NULL;

                ElementType pointValue = dataSource->get(id, divfeat);
                ElementType leafValue = dataSource->get(node->id, divfeat);

                if (pointValue < leafValue) {
                    left->id = id;
                    right->id = node->id;
                }
                else {
                    left->id = node->id;
                    right->id = id;
                }

                left->divfeat = right->divfeat = -1;

                node->divfeat = divfeat;
                node->divval = (pointValue + leafValue) / 2;
                node->child1 = left;
                node->child2 = right;

                // incrementally update imbalance      
                tree->setInsertionLog(id, 0, depth + 2);
                tree->incrementFreqByOne(id);
                tree->incrementFreqAndDepthByOne(nodeId);
            }
            else {
                if (dataSource->get(id, node->divfeat) < node->divval) {
                    addPointToTree(tree, node->child1, id, depth + 1);
                }
                else {
                    addPointToTree(tree, node->child2, id, depth + 1);
                }
            }
        }

        void freeIndex() {
            for (size_t i = 0; i < numTrees; ++i) {
                if (trees[i] != nullptr) trees[i]->~KDTree();
            }
            pool.free();
        }

    public:
        float getMaxCachedCost() {
            float cost = 0;
            for (size_t i = 0; i < numTrees; ++i) {
                if (cost < trees[i]->getCachedCost()) {
                    cost = trees[i]->getCachedCost();
                }
            }

            return cost;
        }

        std::vector<float> getCachedImbalances() {
            std::vector<float> imbalances;
            for (size_t i = 0; i < numTrees; ++i) {
                imbalances.push_back(trees[i]->getCachedImbalance());
            }
            return imbalances;
        }

        std::vector<float> recomputeImbalances() {
            std::vector<float> imbalances;

            for (size_t i = 0; i < numTrees; ++i) {
                imbalances.push_back(trees[i]->computeImbalance());
            }

            return imbalances;
        }

        size_t computeMaxDepth() {
            size_t maxDepth = 0;
            for (size_t j = 0; j < numTrees; ++j) {
                size_t depth = trees[j]->computeMaxDepth();
                if (maxDepth < depth)
                    maxDepth = depth;
            }
            return maxDepth;
        }

        void printBackstage() {
            std::cout << "queue size: " << queue.size() << std::endl;
            std::cout << "ongoingTree size: " << ongoingTree->size << std::endl;
        }

    public:
        UpdateStatus updateStatus = UpdateStatus::NoUpdate;
        KDTree<NodePtr>* ongoingTree;
        float queryLoss = 0.0;
        TreeWeight weight;

    private:
        float reconstructionWeight; // lower => more update

        size_t sizeAtUpdate = 0;

        std::queue<NodeSplit> queue;
        std::vector<size_t> ids;
    };

}

#endif
