#ifndef panene_progressive_knn_table_h
#define panene_progressive_knn_table_h

#include <vector>
#include <iostream>
#include <queue>
#include <set>

#include <progressive_kd_tree_index.h>
#include <functional>

#ifdef BENCHMARK
#include <util/timer.h>
#define BENCH(x) x
#else
#define BENCH(x) ((void)0)
#endif

namespace panene {

    struct TableWeight {
        float treeWeight;
        float tableWeight;

        TableWeight(float treeWeight_, float tableWeight_) : treeWeight(treeWeight_), tableWeight(tableWeight_) {
        }
    };

    struct UpdateResult {
        size_t addPointOps;
        size_t updateIndexOps;
        size_t updateTableOps;

        size_t addPointResult;
        size_t updateIndexResult;
        size_t updateTableResult;

        size_t numPointsInserted;

        double addPointElapsed;
        double updateIndexElapsed;
        double updateTableElapsed;

        std::set<size_t> updatedIds;

        UpdateResult(size_t addPointOps_, size_t updateIndexOps_, size_t updateTableOps_,
            size_t addPointResult_, size_t updateIndexResult_, size_t updateTableResult_,
            size_t numPointsInserted_,
            double addPointElapsed_, double updateIndexElapsed_, double updateTableElapsed_,
            std::set<size_t> updatedIds_
        ) :
            addPointOps(addPointOps_), updateIndexOps(updateIndexOps_), updateTableOps(updateTableOps_),
            addPointResult(addPointResult_), updateIndexResult(updateIndexResult_), updateTableResult(updateTableResult_),
            numPointsInserted(numPointsInserted_),
            addPointElapsed(addPointElapsed_), updateIndexElapsed(updateIndexElapsed_), updateTableElapsed(updateTableElapsed_),
            updatedIds(updatedIds_)
        {
        }
        UpdateResult() = default;
        UpdateResult(const UpdateResult&) = default;

        friend std::ostream& operator<<(std::ostream& os, const UpdateResult& obj) {
            os << "UpdateResult(addPointOps: " << obj.addPointResult << " / " << obj.addPointOps << ", "
                << "updateIndexOps: " << obj.updateIndexResult << " / " << obj.updateIndexOps << ", "
                << "updateTableOps: " << obj.updateTableResult << " / " << obj.updateTableOps << ", numPointsInserted: " << obj.numPointsInserted << ")";
            return os;
        }
    };

    template <typename Indexer, typename DataSink>
    class ProgressiveKNNTable {
        typedef typename Indexer::DataSourceT DataSource;
        typedef typename DataSource::ElementType ElementType;
        typedef typename DataSource::DistanceType DistanceType;
        typedef typename DataSource::IDType IDType;

        typedef Neighbor<IDType, DistanceType> NeighborType;

    public:

        ProgressiveKNNTable(DataSource *dataSource_, DataSink *dataSink_, size_t k_, IndexParams indexParams_, SearchParams searchParams_, TreeWeight treeWeight_, TableWeight weight_) :
            dataSource(dataSource_), dataSink(dataSink_),
            k(k_), weight(weight_), searchParams(searchParams_)
        {
            numPointsInserted = 0;
            indexer = new Indexer(dataSource_, indexParams_, treeWeight_);
            d = dataSource->dim();
            queued = DynamicBitset(dataSource->capacity());
        }

        ProgressiveKNNTable(Indexer *indexer_, DataSink *dataSink_, size_t k_, SearchParams searchParams_, TableWeight weight_) :
            indexer(indexer_), dataSink(dataSink_), k(k_), weight(weight_), searchParams(searchParams_)
        {
            numPointsInserted = 0;
            dataSource = indexer->dataSource;
            d = dataSource->dim();
            queued = DynamicBitset(dataSource->capacity());
        }

        size_t getSize() {
            return indexer->getSize();
        }

        /*
        maxOps indicates the maximum number of operations that the function 'update' can execute.
        There are three types of operations:
          1) addNewPointOp adds a new point P to both table and index. It requires:
            An insertion operation to the index (O(lg N))
            An insertion operation to the table (O(1))
            A knn search for P (O(klg(N)))
            Mark the neighbors of P as dirty and insert them to the queue

          2) updateIndexOp updates a progressive k-d tree index.
            Basically, it calls the update function of the k-d tree and the update function
            creates a new k-d tree incrementally behind the scene

          2) updateTableOp takes a dirty point from a queue and updates its neighbors. It requires:
            A knn search for P
            Mark the neighbors of P as dirty and insert them to the queue


        Note that the three operations have different costs.
        */

        UpdateResult run(size_t ops) {
            // To avoid keeping a copy of the whole data, we need to use an abstract dataframe that grows in real time.

            // 1. add new points to both index and table (addPointOps)
            // 2. update the index (updateIndexOps)

            size_t updateTableOps = 0;
            size_t updateTableResult = 0;
            double updateTableElapsed = 0;

            size_t treeOps = ops * weight.treeWeight;

            auto indexerResult = indexer->run(treeOps);
            size_t addPointResult = indexerResult.addPointResult;

            numPointsInserted += addPointResult;

            size_t size = indexer->getSize();
            size_t oldSize = size - addPointResult;

            // checks if at least one point is added. 
            if (addPointResult > 0) {
                std::vector<IDType> newPoints(addPointResult);

                for (size_t i = 0; i < addPointResult; ++i) {
                    newPoints[i] = oldSize + i;
                }

                std::vector<ResultSet<IDType, DistanceType>> results(addPointResult);
                for (size_t i = 0; i < addPointResult; ++i)
                    results[i] = ResultSet<IDType, DistanceType>(k);

                indexer->knnSearch(newPoints, results, k, searchParams);

                for (size_t i = 0; i < addPointResult; ++i) {
                    IDType id = oldSize + i;
                    queued.set(id);

                    for (IDType j = 0; j < k; ++j) {
                        if (!queued.test(results[i][j].id)) {
                            queued.set(results[i][j].id);
                            queue.push(results[i][j]);
                        }
                    }

                    dataSink->setNeighbors(id,
                        results[i].getNeighbors(),
                        results[i].getDistances());
                }
            }


            // 3. process the queue (updateTableOps)
            updateTableOps = ops - indexerResult.addPointOps - indexerResult.updateIndexOps;

            BENCH(Timer timer);
            BENCH(timer.begin());

            size_t checkCount = 0;
            std::set<IDType> updatedIds;

            while (checkCount < updateTableOps && !queue.empty()) {
                auto q = queue.top();

                queue.pop();
                checkCount++;

                // we need to update the NN of q.id

                // get the new NN of the dirty point
                ResultSet<IDType, DistanceType> result(k);
                indexer->knnSearch(q.id, result, k, searchParams);

                // check if there is a difference between previous NN and newly computed NN.      
                size_t i;
                std::vector<IDType> current(k);
                dataSink->getNeighbors(q.id, current);
                for (i = 0; i < k; ++i) {
                    if (current[i] != result[i].id)
                        break;
                }

                if (i < k) { // if there is a difference
                  // then, mark the nn of q.id as dirty

                    updatedIds.insert(q.id);
                    dataSink->setNeighbors(
                        q.id,
                        result.getNeighbors(),
                        result.getDistances());

                    for (i = 0; i < k; ++i) {
                        if (!queued.test(result[i].id)) {
                            queued.set(result[i].id);
                            queue.push(result[i]);
                        }
                    }
                }

                queued.reset(q.id);
            }

            updateTableResult = checkCount;
            BENCH(updateTableElapsed = timer.end());

            return UpdateResult(indexerResult.addPointOps, indexerResult.updateIndexOps, updateTableOps,
                addPointResult, indexerResult.updateIndexResult, updateTableResult,
                numPointsInserted,
                indexerResult.addPointElapsed, indexerResult.updateIndexElapsed, updateTableElapsed,
                updatedIds);
        }

        UpdateResult runWithoutIndexer(UpdateResult2 indexerResult, size_t updateTableOps) {
            // 1. add new points to both index and table (addPointOps)
            // 2. update the index (updateIndexOps)

            size_t addPointResult = indexerResult.addPointResult;
            size_t updateTableResult = 0;
            double updateTableElapsed = 0;

            numPointsInserted += addPointResult;

            size_t size = indexer->getSize();
            size_t oldSize = size - addPointResult;

            // checks if at least one point is added. 
            if (addPointResult > 0) {
                std::vector<IDType> newPoints(addPointResult);

                for (size_t i = 0; i < addPointResult; ++i) {
                    newPoints[i] = oldSize + i;
                }

                std::vector<ResultSet<IDType, DistanceType>> results(addPointResult);
                for (size_t i = 0; i < addPointResult; ++i)
                    results[i] = ResultSet<IDType, DistanceType>(k);

                indexer->knnSearch(newPoints, results, k, searchParams);

                for (size_t i = 0; i < addPointResult; ++i) {
                    IDType id = oldSize + i;
                    queued.set(id);

                    for (IDType j = 0; j < k; ++j) {
                        if (!queued.test(results[i][j].id)) {
                            queued.set(results[i][j].id);
                            queue.push(results[i][j]);
                        }
                    }

                    dataSink->setNeighbors(id,
                        results[i].getNeighbors(),
                        results[i].getDistances());
                }
            }


            // 3. process the queue (updateTableOps)
            BENCH(Timer timer);
            BENCH(timer.begin());

            size_t checkCount = 0;
            std::set<IDType> updatedIds;

            while (checkCount < updateTableOps && !queue.empty()) {
                auto q = queue.top();

                queue.pop();
                checkCount++;

                // we need to update the NN of q.id

                // get the new NN of the dirty point
                ResultSet<IDType, DistanceType> result(k);
                indexer->knnSearch(q.id, result, k, searchParams);

                // check if there is a difference between previous NN and newly computed NN.      
                size_t i;
                std::vector<IDType> current(k);
                dataSink->getNeighbors(q.id, current);
                for (i = 0; i < k; ++i) {
                    if (current[i] != result[i].id)
                        break;
                }

                if (i < k) { // if there is a difference
                  // then, mark the nn of q.id as dirty

                    updatedIds.insert(q.id);
                    dataSink->setNeighbors(
                        q.id,
                        result.getNeighbors(),
                        result.getDistances());

                    for (i = 0; i < k; ++i) {
                        if (!queued.test(result[i].id)) {
                            queued.set(result[i].id);
                            queue.push(result[i]);
                        }
                    }
                }

                queued.reset(q.id);
            }

            updateTableResult = checkCount;
            BENCH(updateTableElapsed = timer.end());

            return UpdateResult(indexerResult.addPointOps, indexerResult.updateIndexOps, updateTableOps,
                addPointResult, indexerResult.updateIndexResult, updateTableResult,
                numPointsInserted,
                indexerResult.addPointElapsed, indexerResult.updateIndexElapsed, updateTableElapsed,
                updatedIds);
        }



        const void getNeighbors(const IDType id, std::vector<IDType> &res) const {
            dataSink->getNeighbors(id, res);
        }

        const void getDistances(const IDType id, std::vector<DistanceType> &res) const {
            dataSink->getDistances(id, res);
        }

    private:
        DataSource * dataSource;
        DataSink *dataSink;
        size_t d;
        size_t k;

    public:
        Indexer * indexer;

    private:
        TableWeight weight;
        SearchParams searchParams;
        size_t numPointsInserted;

        std::priority_queue<NeighborType, std::vector<NeighborType>, std::greater<NeighborType>> queue; // descending order
        DynamicBitset queued;
    };

}
#endif
