#ifndef TRAJECTORY_HPP
#define TRAJECTORY_HPP

#include <set>
#include <boost/serialization/list.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/vector.hpp>
#include <flann/flann.hpp>

#include "bib/Assert.hpp"
#include "bib/Logger.hpp"
#include "bib/Utils.hpp"

inline bool less_float_cutted(float a, float b){
    return a < b;
}

inline bool diff_float_cutted(float a, float b){
    return a != b;
}

// inline bool less_float_cutted(float a, float b, uint precision=3) { //4 -> 0.0001
//   float aa = a*pow(10,precision);
//   float bb = b*pow(10,precision);
//   int aaa = (int)aa;
//   int bbb = (int) bb;
//   return aaa < bbb;
// }
// 
// inline bool diff_float_cutted(float a, float b, uint precision=3) { //4 -> 0.0001
//   float aa = a*pow(10,precision);
//   float bb = b*pow(10,precision);
//   int aaa = (int)aa;
//   int bbb = (int) bb;
//   return aaa != bbb;
// }


typedef struct _sa_sample {
  std::vector<float> s;
  std::vector<float> a;

  friend class boost::serialization::access;
  template <typename Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar& BOOST_SERIALIZATION_NVP(s);
    ar& BOOST_SERIALIZATION_NVP(a);
  }

  bool operator< (const _sa_sample& b) const {
    for (uint i = 0; i < s.size(); i++) {
      if(diff_float_cutted(s[i], b.s[i]))
        return less_float_cutted(s[i] , b.s[i]);
    }
//     for (uint i = 0; i < a.size(); i++) {
//       if(diff_float_cutted(a[i] , b.a[i]))
//         return less_float_cutted(a[i] , b.a[i]);
//     }

    return false;
  }

  const float* data_kd_tree() const {
    return s.data();
  }

} SA_sample;

typedef struct _sasrg_sample : public SA_sample {
  std::vector<float> next_s;
  double r;
  bool goal_reached;

  friend class boost::serialization::access;
  template <typename Archive>
  void serialize(Archive& ar, const unsigned int v) {
    SA_sample::serialize(ar, v);
    LOG_DEBUG("check I can do that");
    ar& BOOST_SERIALIZATION_NVP(next_s);
    ar& BOOST_SERIALIZATION_NVP(r);
    ar& BOOST_SERIALIZATION_NVP(goal_reached);
  }

  bool operator< (const _sasrg_sample& b) const {
    for (uint i = 0; i < s.size(); i++) {
      if(diff_float_cutted(s[i], b.s[i]))
        return less_float_cutted(s[i] , b.s[i]);
    }
    for (uint i = 0; i < a.size(); i++) {
      if(diff_float_cutted(a[i] , b.a[i]))
        return less_float_cutted(a[i] , b.a[i]);
    }

    for (uint i = 0; i < next_s.size(); i++) {
      if(diff_float_cutted(next_s[i] , b.next_s[i]))
        return less_float_cutted(next_s[i] , b.next_s[i]);
    }

    if(diff_float_cutted(r , b.r))
      return less_float_cutted(r , b.r);

    return less_float_cutted(goal_reached , b.goal_reached);
  }

} SASRG_sample;


//  Distance=flann::L2<float>
template <typename Struct=SA_sample, typename Distance=flann::L1<float> >
class Trajectory {

 public:
  Trajectory(size_t _column):column(_column) {}
  
  ~Trajectory(){
      delete kd_tree;
  }

  size_t size() {
    ASSERT(kd_tree == nullptr || tree_trajectory.size() == kd_tree->size(), "unconsistency");

    return tree_trajectory.size();
  }

  void addPoint(Struct& elem) {

    auto pr = tree_trajectory.insert(elem);
    if(!pr.second) { //already in -> point very near of another : keep or replace it?
      //better to replace to not cause pointer problem inside kd-tree
      remove_point(pr.first->data_kd_tree());
      tree_trajectory.erase(pr.first);
      pr = tree_trajectory.insert(elem);
      ASSERT(pr.second, "unconsistency");
    }
    
    //look for the closest and may be remove it
    flann::Matrix<float> kas((float*)pr.first->data_kd_tree(), 1, column);

    if(kd_tree != nullptr) {

      flann::KNNResultSet<float> result(std::min((size_t)3, kd_tree->size()));
//       LOG_DEBUG(kas.ptr()[0] << " " << kas.ptr()[1] << " " << kas.ptr()[2] );
      kd_tree->buildIndex(); //mandatory
      kd_tree->findNeighbors(result, kas.ptr(), flann::SearchParams(-1));
      uint nn = result.size();
      size_t * indices = new size_t[nn];
      float * dists = new float[nn];
      result.copy(indices, dists, nn);

      for (uint i=0;i<nn;i++){
//         float distance = dists[i] / (2. * column); // in [0;1]
//         distance = 1.f - pow(2*distance*column, 0.1); // 1-\left(2\cdot 3\cdot x\right)^{0.1}
        float distance = 1.f - pow(2*dists[i], 0.1); //small than 0.5 -> implies probability to drop

        if(distance > 0 && bib::Utils::rand01() < distance) {
//             if(kd_tree->getPoint(indices[i]) == 0){
//                 LOG_DEBUG("remove proba(" << distance << ") vec dist(" << dists[i] << ") normal dist(" << dists[i] / (2. * column) << " " << indices[i]);
//                 bib::Logger::PRINT_ELEMENTS<>(kas.ptr(), 3);
//                 bib::Logger::PRINT_ELEMENTS<>(kd_tree->getPoint(indices[i]), 3);
//             }
            remove_index(indices[i]);
        }
      }

      delete[] indices;
      delete[] dists;

      kd_tree->buildIndex();
      kd_tree->addPoints(kas);
      
//       if(exists_null_ptr()){
//             bib::Logger::PRINT_ELEMENTS<>(kas.ptr(), 3);
//             bib::Logger::PRINT_ELEMENTS<>(kd_tree->getPoint(kd_tree->size()), 3);
//             bib::Logger::PRINT_ELEMENTS<>(removed_index);
//       }
//       ASSERT(!exists_null_ptr(), "null ptr2");
    } else{
//       flann::KDTreeSingleIndexParams parms;
//       parms.clear();//let it take the default params, unless valgrind crying        
//       kd_tree = new flann::KDTreeSingleIndex<Distance >(kas, parms);
      flann::KDTreeSingleIndexParams parms;
      parms["trees"] = 1;
      kd_tree = new flann::KDTreeIndex<Distance >(kas, parms);
    }

    LOG_DEBUG(tree_trajectory.size() << " " << kd_tree->size());
    ASSERT(tree_trajectory.size() == kd_tree->size(), "unconsistency " << tree_trajectory.size() << " vs "<< kd_tree->size());
  }
  
  void print(){
       for(size_t i=0;i < kd_tree->size();i++){
            float* p = kd_tree->getPoint(i);
            std::cout << i << " : (";
            for(uint j=0; j < column;j++)
                std::cout << p[j] << ", ";
            std::cout << ") "<<  std::endl;
    }
    std::cout << std::endl;   
  }
  
protected:
  
    void remove_index(size_t index){
        float* pp = kd_tree->getPoint(index);
            
        std::vector<float> cm(column);
        
        for (size_t i=0; i<column; i++)
                cm[i]=pp[i];
        
        Struct elem_rm;
        elem_rm.s = cm;
            
        kd_tree->removePoint(index);
        
        auto pr = tree_trajectory.find(elem_rm);
        ASSERT(pr != tree_trajectory.end(), "error unconsistency");
        tree_trajectory.erase(pr);
    }
    
    void remove_point(const float* tab){
        flann::Matrix<float> kas((float*)tab, 1, column);
        flann::KNNResultSet<float> result(1);
        kd_tree->buildIndex();
        kd_tree->findNeighbors(result, kas.ptr(), flann::SearchParams(-1));
        uint nn = result.size();
        size_t * indices = new size_t[nn*column];
        float * dists = new float[nn*column];
        result.copy(indices, dists, nn);
        ASSERT(nn == 1 && dists[0] == 0, "error unconsistency");
        kd_tree->removePoint(indices[0]);
        kd_tree->buildIndex();
        
        delete[] indices;
        delete[] dists;
    }

 private :
  size_t column;

  std::set<Struct> tree_trajectory;
//   flann::KDTreeSingleIndex<Distance >* kd_tree = nullptr;
  flann::KDTreeIndex<Distance >* kd_tree = nullptr;
  
  //         flann::Matrix<float> empty_dataset(new float[0], 0, nb_sensors);
//      KDTreeSingleIndex is efficient for low dimensional data -> exact NN search
//      KDTreeIndexParams(1) high dimensional data -> approximative search
//      flann::KDTreeIndex   kd_tree = new flann::Index<flann::L2<float> >(empty_dataset, flann::KDTreeIndexParams(1));
};

#endif
