#ifndef DUMPER_H
#define DUMPER_H

#include <tuple>

namespace helper {
template <int... Is>
struct index {};

template <int N, int... Is>
struct gen_seq : gen_seq < N - 1, N - 1, Is... > {};

template <int... Is>
struct gen_seq<0, Is...> : index<Is...> {};
}
namespace bib {

template <typename Displayable, typename... Args>
class Dumper {
 public:
  Dumper(Displayable* _ptr, Args... _args) : ptr(_ptr), params(_args...) {}
  template <int... Is>
  std::ostream& deploy(std::ostream& out, helper::index<Is...>) const {
    return ptr->display(out, std::get<Is>(params)...);
  }
  friend std::ostream& operator<<(std::ostream& out, const Dumper& disp) {
    return disp.deploy(out, helper::gen_seq<sizeof...(Args)> {});
  }

 private:
  Displayable* ptr;
  std::tuple<Args...> params;
};
} // namespace bib

#endif  // DUMPER_H
