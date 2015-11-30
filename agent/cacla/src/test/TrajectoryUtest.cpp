#include "gtest/gtest.h"
#include "Trajectory.hpp"
#include "bib/Logger.hpp"
#include "kdtree++/kdtree.hpp"

TEST(Trajectory, LowLevel){
    typedef KDTree::KDTree<SA_sample, KDTree::_Bracket_accessor<SA_sample>, L1_distance<SA_sample::value_type, SA_sample::value_type> > tree_type;

    SA_sample x1 = {{0.5, -0.1},{1}};
    SA_sample x2 = {{0.5, 0.1},{1}};
    SA_sample x3 = {{0.9, -0.9},{1}};
    SA_sample x4 = {{0.7, -0.8},{1}};
    
    tree_type t(2);
    t.insert(x1);
    t.insert(x2);
    t.insert(x3);
    
    auto pr = t.find_nearest(x4);
    LOG_DEBUG(pr.second*pr.second << " " << pr.first->s[0]<< " " << pr.first->s[1]);

    t.erase(pr.first);
    t.insert(x4);
    
    for(auto it = t.begin();it != t.end();++it)
        LOG_DEBUG(it->s[0] << " " << it->s[1]);
    
    EXPECT_EQ(t.size(), 3);
}

TEST(Trajectory, AddedWell) {
    for(uint i=0;i< 500; i++){
        Trajectory<SA_sample> t(3);
        SA_sample sa1 = {{1,0.5,-1},{}};
        SA_sample sa2 = {{0.5,0.5,-1},{}};
        SA_sample sa3 = {{0.5,0.5,1},{}};
        SA_sample sa4 = {{1,-0.5,1},{}};
        t.addPoint(sa1);
        t.addPoint(sa2);
        t.addPoint(sa3);
        t.addPoint(sa4);
        EXPECT_EQ(t.size(), 4);
    }
}


TEST(Trajectory, RemoveByAddition) {
    uint keeped= 0 ;
    uint removed= 0 ;
    
    for(uint i=0;i< 500; i++){
        Trajectory<> t(3);
        SA_sample sa1 = {{1,0.5,-1},{}};
        SA_sample sa2 = {{0.5,0.5,-1},{}};
        SA_sample sa3 = {{0.5,0.5,1},{}};
        SA_sample sa4 = {{1,0.5,-0.8},{}};
        t.addPoint(sa1);
        t.addPoint(sa2);
        t.addPoint(sa3);
        t.addPoint(sa4);
        
        bool too_near_stochas = t.size() == 3 || t.size() == 4; 
        if(t.size() == 4)
            keeped++;
        else
            removed++;
        EXPECT_EQ(too_near_stochas, true);
    }
    
    EXPECT_GT(keeped, 0);
    EXPECT_GT(removed, 0);
    LOG_DEBUG(keeped << " " << removed);
}

TEST(Trajectory, Collision){
    for(uint i=0;i< 500; i++){
        Trajectory<> t(3);
        SA_sample sa1 = {{1,0.5,-1},{}};
        SA_sample sa2 = {{0.5,0.5,-1},{}};
        SA_sample sa3 = {{0.5,0.5,1},{}};
        SA_sample sa4 = {{1,0.5,-0.99999999999},{}};
        t.addPoint(sa1);
        t.addPoint(sa2);
        t.addPoint(sa3);
        t.addPoint(sa4);
        
        EXPECT_EQ(t.size(), 3);
    }
}

// OLD BUGGED FLANN
// TEST(Trajectory, RemoveAndFindLL){
//     
//     std::vector<double> x1({1,0.5}); //this point is going to be removed
//     std::vector<double> x2({0.8,0.5}); //this point should be found by NN
//     std::vector<double> x3({0.99,0.5}); // looking for the closest
//     
//     flann::Matrix<double> mx1(x1.data(), 1, 2);
//     flann::Matrix<double> mx2(x2.data(), 1, 2);
//     flann::Matrix<double> mx3(x3.data(), 1, 2);
//     
//     //build single tree with x1
//     flann::IndexParams params;
//     params["trees"] = 1;
// //     params["reorder"] = false;
//     flann::KDTreeIndex< flann::L1<double> > kd_tree(mx1, params);
// //     flann::KDTreeSingleIndex< flann::L1<double> > kd_tree(mx1, params);
//     kd_tree.addPoints(mx2);
//     
//     //display x1 and x2 with their index
//     for(size_t i=0;i < kd_tree.size();i++){
//             double* p = kd_tree.getPoint(i);
//             std::cout << i << " : ("  << p[0] << ", " << p[1] << ") "<<  std::endl;
//     }std::cout << std::endl;
//     
//     //remove point x1
//     kd_tree.removePoint(0);
//     kd_tree.buildIndex();
//     
//     //look for nearest point of x3 : good result is x2 because x1 has been removed
//     flann::KNNResultSet<double> result(1);
//     kd_tree.buildIndex(); 
//     kd_tree.findNeighbors(result, mx3.ptr(), flann::SearchParams(-1));
//     size_t* indices = new size_t[1];
//     double* dists = new double[1];
//     result.copy(indices, dists, 1);
//     
//     std::cout << indices[0] << " " << dists[0] << std::endl;
//     std::cout << kd_tree.getPoint(indices[0]) << std::endl;
//     std::cout << kd_tree.getPoint(indices[0])[0] << " " << kd_tree.getPoint(indices[0])[1] << std::endl;
//     
//     EXPECT_EQ(kd_tree.getPoint(indices[0])[0], x2[0]);
//     EXPECT_EQ(kd_tree.getPoint(indices[0])[1], x2[1]);
//     
//     delete[] indices;
//     delete[] dists;
// }

TEST(Trajectory, RemoveAndFind){  
    for(uint i=0;i< 500; i++){
        Trajectory<> t(3);
        SA_sample sa1 = {{1,0.5,-1},{}};
        SA_sample sa2 = {{0.998,0.5,-1},{}};
        SA_sample sa3 = {{0.99999,0.5,-1},{}};
        t.addPoint(sa1);
        t.addPoint(sa2);
        t.addPoint(sa3);

        t.print();
    }
}

TEST(Trajectory, RandomInsertion){
    Trajectory<> t(3);
    
    for(uint i=0;i< 500; i++){
        double x1 = bib::Utils::rand01()*2 - 1;
        double x2 = bib::Utils::rand01()*2 - 1;
        double x3 = bib::Utils::rand01()*2 - 1;
        
        SA_sample sa1 = {{x1, x2, x3},{}};
        t.addPoint(sa1);
    }
    
    LOG_DEBUG(t.size());
}