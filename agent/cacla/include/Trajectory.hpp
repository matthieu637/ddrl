#ifndef TRAJECTORY_HPP
#define TRAJECTORY_HPP

#include <set>
#include <boost/serialization/list.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/vector.hpp>

#include "bib/Assert.hpp"
#include "bib/Logger.hpp"
#include "bib/Utils.hpp"
#include "kdtree++/kdtree.hpp"

#define KDTREE_OWNDIST

template <typename _Tp, typename _Dist>
struct L1_distance
{
  typedef _Dist distance_type;

  distance_type
  operator() (const _Tp& __a, const _Tp& __b, const size_t) const
  {
    distance_type d = fabs(__a - __b);
    return d;
  }
};

typedef struct _sa_sample {
  std::vector<double> s;
  std::vector<double> a;
  
  typedef double value_type;

  friend class boost::serialization::access;
  template <typename Archive>
  void serialize(Archive& ar, const unsigned int) {
    ar& BOOST_SERIALIZATION_NVP(s);
    ar& BOOST_SERIALIZATION_NVP(a);
  }

  bool operator< (const _sa_sample& b) const {
    for (uint i = 0; i < s.size(); i++) {
      if(s[i] != b.s[i])
        return s[i] < b.s[i];
    }

    return false;
  }

  inline value_type operator[](size_t const N) const{
        return s[N];
  }

} SA_sample;

typedef struct _sasrg_sample {
  std::vector<double> s;
  std::vector<double> a;
  std::vector<double> next_s;
  double r;
  bool goal_reached;
  
  typedef double value_type;

  friend class boost::serialization::access;
  template <typename Archive>
  void serialize(Archive& ar, const unsigned int v) {
    ar& BOOST_SERIALIZATION_NVP(s);
    ar& BOOST_SERIALIZATION_NVP(a);
    ar& BOOST_SERIALIZATION_NVP(next_s);
    ar& BOOST_SERIALIZATION_NVP(r);
    ar& BOOST_SERIALIZATION_NVP(goal_reached);
  }

  bool operator< (const _sasrg_sample& b) const {
    for (uint i = 0; i < s.size(); i++) {
      if(s[i] != b.s[i])
        return s[i] < b.s[i];
    }
    
    return false;
  }
    
  inline double operator[](size_t const N) const{
        return s[N];
  }
  
} SASRG_sample;


typedef struct _qsasrg_sample {
  std::vector<double> s;
  std::vector<double> a;
  std::vector<double> next_s;
  double r;
  bool goal_reached;
  
  typedef double value_type;

  friend class boost::serialization::access;
  template <typename Archive>
  void serialize(Archive& ar, const unsigned int v) {
    ar& BOOST_SERIALIZATION_NVP(s);
    ar& BOOST_SERIALIZATION_NVP(a);
    ar& BOOST_SERIALIZATION_NVP(next_s);
    ar& BOOST_SERIALIZATION_NVP(r);
    ar& BOOST_SERIALIZATION_NVP(goal_reached);
  }

  bool operator< (const _qsasrg_sample& b) const {
    for (uint i = 0; i < s.size(); i++) {
      if(s[i] != b.s[i])
        return s[i] < b.s[i];
    }
    
    for (uint i = 0; i < a.size(); i++) {
      if(a[i] != b.a[i])
        return a[i] < b.a[i];
    }
    
    return false;
  }
    
  inline double operator[](size_t const N) const{
        size_t l = s.size();
        return N < l ? s[N] : a[N-l];
  }
  
} QSASRG_sample;


template<typename Struct>
struct UniqTest
{
   bool operator()(const Struct& t ) const
   {
      return been_tested.find(t) == been_tested.end();//not in => not yet tested
   }
   
   std::set<Struct> been_tested;
};

class Observer{
public:
  Observer(uint _size) : column(_size), data_number(0), mean(_size, 0), var(_size, 0), subvar(_size, 0){}
  
  void transform(std::vector<double>& output, const std::vector<double> x){
    update_mean_var(x);
  
    for(uint i=0; i < column; i++){
        output[i] = (x[i] - mean[i]);

        if(var[i] != 0.d)
          output[i] /= sqrt(var[i]) / 0.26115f;
    }

  }
  
  void update_mean_var(const std::vector<double>& x){
    double dndata_number = data_number + 1;
    double ddata_number = data_number;
    for(uint i=0; i < column; i++){
      mean[i] = (mean[i] * ddata_number + x[i])/dndata_number;
    
      subvar[i] = (subvar[i] * ddata_number + (x[i]*x[i]))/dndata_number;
      var[i] = subvar[i] - (mean[i]*mean[i]);
    }
    
    data_number ++;
  }
  
  uint column;
  ulong data_number;
  std::vector<double> mean;
  std::vector<double> var;
  std::vector<double> subvar;
};

template <typename _Tp, typename _Dist>
class ObserverL1Distance : public Observer{
public:
  typedef _Dist distance_type;
  
  ObserverL1Distance(uint _size):Observer(_size){}

  distance_type
  operator() (const _Tp& __a, const _Tp& __b, const size_t __dim) const
  {
    distance_type d = fabs(__a - __b);
    if(data_number > 3)
      d /= (sqrt(var[__dim])/0.26115d);
    
    return d;
  }
};

template <typename Struct=SA_sample >
class Trajectory {

#ifdef KDTREE_OWNDIST
  typedef KDTree::KDTree<Struct, KDTree::_Bracket_accessor<Struct>, ObserverL1Distance< typename Struct::value_type, typename Struct::value_type> > tree_type;
#else
  typedef KDTree::KDTree<Struct, KDTree::_Bracket_accessor<Struct>, L1_distance< typename Struct::value_type, typename Struct::value_type> > tree_type;
#endif
  
 public:
  Trajectory(size_t _column, double _proba_transform=0.5f, bool _remove_nearest=true, double _proba_threshold=1.f): 
  column(_column), 
#ifdef KDTREE_OWNDIST
  obs(_column),
  kd_tree(_column, KDTree::_Bracket_accessor<Struct>(), obs),
#else
  kd_tree(_column),
#endif
  proba_transform(_proba_transform), remove_nearest(_remove_nearest), proba_threshold(_proba_threshold), count_removed(0), count_inserted(0) {}
  
  ~Trajectory(){

  }

  size_t size() {
    return kd_tree.size();
  }

  void addPoint(Struct& elem) {
    count_inserted++;
    
#ifdef KDTREE_OWNDIST
    std::vector<double> x(column);
    for(uint i=0; i < column;i++)
      x[i] = elem[i];
    obs.update_mean_var(x);
#endif

    if(remove_nearest){
      UniqTest<Struct> predicate;
      
      while(kd_tree.size() - predicate.been_tested.size() > 0){
          //look for the closest and may be remove it
//           auto pr = kd_tree.find_nearest_if(elem, 2.f * column, predicate);
//           double proba = 1.f - pow(pr.second/proba_threshold, proba_transform); //small than 0.5 -> implies probability to drop
          
          auto pr = kd_tree.find_nearest_if(elem, sqrt(column), predicate);
          double proba = 1.f - powf((pr.second*pr.second)/(column), proba_transform);
          
//           LOG_DEBUG(proba << " " << (pr.second*pr.second)/column );
//           bib::Logger::PRINT_ELEMENTS<>(*pr.first, 4);
//           bib::Logger::PRINT_ELEMENTS<>(elem.s, 4);
          
//           ASSERT(pr.second <= 2 * column, "not normalized");

          if(proba > 0 && bib::Utils::rand01() < proba) {
  //             if(kd_tree->getPoint(indices[i]) == 0){
  //                 LOG_DEBUG("remove proba(" << distance << ") vec dist(" << dists[i] << ") normal dist(" << dists[i] / (2. * column) << " " << indices[i]);
  //                 bib::Logger::PRINT_ELEMENTS<>(kd_tree->getPoint(indices[i]), 3);
  //             }
              kd_tree.erase(pr.first);
              count_removed++;
          }
          else if(proba <= 0)
            break;
          else //proba >= 0 but didn't been tackle
            predicate.been_tested.insert(*pr.first);
      }
    }
    
      //if elem colide with a data, it has just been removed with probability 1
      kd_tree.insert(elem);
    
  }
  
  tree_type& tree(){
      return kd_tree;
  }
  
  void print(){
//     for(auto it = kd_tree.begin(); it != kd_tree.end() ; ++it)
//       LOG_DEBUG(it->print);
  }
  void clear(){
      kd_tree.clear();
  }
  
 private :
  size_t column;
#ifdef KDTREE_OWNDIST
  ObserverL1Distance<typename Struct::value_type, typename Struct::value_type> obs;
#endif
  tree_type kd_tree;
  const double proba_transform;
  bool remove_nearest;
  const double proba_threshold;
public:
  uint count_removed;
  uint count_inserted;
};

#endif
