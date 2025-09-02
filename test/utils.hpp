#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "algo/algo.hpp"
#include "benchUtils.h"
#include "parlay.hpp"
// #include "cpam.hpp"
#include "dist.hpp"
#include "graph/adj.hpp"
#include "parse_points.hpp"
#include "util/intrin.hpp"
#include "io/writer.hpp"

#define RED "\033[31m"
#define GREEN "\033[32m"
#define YELLOW "\033[33m"
#define BLUE "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN "\033[36m"
#define RESET "\033[0m"

template<class G>
void print_stat(const G &g) {
  puts("#vertices         edges        avg. deg");
  size_t cnt_vertex = g.num_nodes();
  size_t cnt_degree = g.num_edges();
  printf("%lu %16lu %10.2f\n", cnt_vertex, cnt_degree, float(cnt_degree) / cnt_vertex);
}

template<class G>
void print_layers(const G &g) {
  const uint32_t height = g.get_height();
  printf("Highest level: %u\n", height);
  puts("level     #vertices         edges        avg. deg");
  for (uint32_t i = 0; i <= height; ++i) {
    const uint32_t level = height - i;
    size_t cnt_vertex = g.num_nodes(level);
    size_t cnt_degree = g.num_edges(level);
    printf("#%2u: %14lu %16lu %10.2f\n", level, cnt_vertex, cnt_degree,
           float(cnt_degree) / cnt_vertex);
  }
}

// assume the snapshots are only of increasing insertions
template<class R>
void print_stat_snapshots(const R &snapshots) {
  const auto &fin = snapshots.back();
  std::vector<std::unordered_set<uint64_t>> cnts_vtx(fin.get_height() + 1);
  std::vector<std::unordered_map<uint64_t, size_t>> cnts_edge(fin.get_height() + 1);
  // count the vertices and the edges
  for (const auto &g : snapshots) {
    auto sum_edges = [](const auto &m) {
      return std::transform_reduce(m.begin(), m.end(), size_t(0), std::plus<size_t>{},
                                   // [](uint64_t eid){return size_t(eid&0xffff);}
                                   [](std::pair<uint64_t, size_t> p) {
                                     return p.second;
                                   });
    };

    const uint32_t height = g.get_height();
    printf("Highest level: %u\n", height);
    puts("level     #vertices      edges    avg. deg   ref#vtx   ref#edge");
    for (uint32_t i = 0; i <= height; ++i) {
      const uint32_t l = height - i;
      size_t cnt_vertex = g.num_nodes(l);
      size_t cnt_degree = g.num_edges(l);
      auto refs = g.collect_refs(l);

      // TODO: use std::transform_view in C++20
      for (auto [key, val] : refs) {
        cnts_vtx[l].insert(key);
        uint64_t ptr_el = val >> 16;
        size_t size_el = val & 0xffff;
        auto &old_size = cnts_edge[l][ptr_el];
        if (old_size < size_el) old_size = size_el;
      }

      size_t ref_vtx = cnts_vtx[l].size();
      size_t ref_edge = sum_edges(cnts_edge[l]);
      printf("#%2u: %14lu %10lu %10.2f %10lu %10lu\n", l, cnt_vertex, cnt_degree,
             float(cnt_degree) / cnt_vertex, ref_vtx, ref_edge);
    }
  }
  // sample 100 vertices and observe the historical changes of their edge lists
  std::mt19937 gen(1206);
  std::uniform_int_distribution<uint32_t> distrib(0, fin.num_nodes(0));
  std::vector<uint32_t> samples(100);
  for (uint32_t &sample : samples) {
    sample = distrib(gen);
    printf("%u ", sample);
  }
  putchar('\n');

  for (size_t i = 0; i < snapshots.size(); ++i) {
    const auto &s = snapshots[i];
    for (uint32_t sample : samples) {
      if (sample >= s.num_nodes(0)) {
        // printf("* ");
        printf("0 ");
        continue;
      }
      if (i == 0 || sample >= snapshots[i - 1].num_nodes(0)) {
        // printf("+%lu,-0 ", s.get_nbhs(sample).size());
        printf("%lu ", s.get_nbhs(sample).size());
        continue;
      }

      auto calc_diff = [](auto agent_last, auto agent_curr) {
        using edge_t = typename decltype(agent_last)::value_type;
        auto last = ANN::util::to<parlay::sequence<edge_t>>(agent_last);
        auto curr = ANN::util::to<parlay::sequence<edge_t>>(agent_curr);
        std::sort(last.begin(), last.end());
        std::sort(curr.begin(), curr.end());
        parlay::sequence<edge_t> comm;
        std::set_intersection(last.begin(), last.end(), curr.begin(), curr.end(),
                              std::inserter(comm, comm.end()));
        return std::make_pair(curr.size() - comm.size(), last.size() - comm.size());
      };

      auto diff = calc_diff(snapshots[i - 1].get_nbhs(sample), s.get_nbhs(sample));
      // printf("+%lu,-%lu ", diff.first, diff.second);
      printf("%lu ", diff.first);
    }
    putchar('\n');
  }
}

template<class G, class Seq>
auto find_nbhs(const G &g, const Seq &q, uint32_t k, uint32_t ef) {
  const size_t cnt_query = q.size();
  // per_visited.resize(cnt_query);
  // per_eval.resize(cnt_query);
  // per_size_C.resize(cnt_query);

  using seq_result = parlay::sequence<typename G::result_t>;
  parlay::sequence<seq_result> res(cnt_query);
  auto search = [&] {
    parlay::parallel_for(0, cnt_query, [&](size_t i) {
      ANN::algo::search_control ctrl{};
      //ctrl.log_per_stat = i;
      // ctrl.beta = beta;
      res[i] = g.template search<seq_result>(q[i].get_coord(), k, ef, ctrl);
    });
  };

  // puts("Warmup");
  search();

  parlay::internal::timer t;
  const uint32_t rounds = 5;
  for (uint32_t i = 0; i < rounds; ++i) search();
  const double time_query = t.next_time() / rounds;
  const double qps = cnt_query / time_query;
  printf("Find neighbors: %.4f s, %e kqps\n", time_query, qps / 1000);

  // printf("# visited: %lu\n", parlay::reduce(per_visited, parlay::addm<size_t>{}));
  // printf("# eval: %lu\n", parlay::reduce(per_eval, parlay::addm<size_t>{}));
  // printf("size of C: %lu\n", parlay::reduce(per_size_C, parlay::addm<size_t>{}));

  // parlay::sort_inplace(per_visited);
  // parlay::sort_inplace(per_eval);
  // parlay::sort_inplace(per_size_C);
  // const double tail_ratio[] = {0.9, 0.99, 0.999};
  // for (size_t i = 0; i < sizeof(tail_ratio) / sizeof(*tail_ratio); ++i) {
  //   const auto r = tail_ratio[i];
  //   const uint32_t tail_index = r * cnt_query;
  //   printf("%.4f tail stat (at %u):\n", r, tail_index);

  //   printf("\t# visited: %lu\n", per_visited[tail_index]);
  //   printf("\t# eval: %lu\n", per_eval[tail_index]);
  //   printf("\tsize of C: %lu\n", per_size_C[tail_index]);
  // }

  return res;
}

// For filter
template<class G, class Seq, typename L>
auto find_nbhs(const G &g, const Seq &q, uint32_t k, uint32_t ef,
               const parlay::sequence<parlay::sequence<L>> &F, uint32_t rounds = 5/*, bool filtered*/) {
  const size_t cnt_query = q.size();
  // per_visited.resize(cnt_query);
  // per_eval.resize(cnt_query);
  // per_size_C.resize(cnt_query);

  using seq_result = parlay::sequence<typename G::result_t>;
  parlay::sequence<seq_result> res(cnt_query);
  auto search = [&] {
    parlay::parallel_for(0, cnt_query, [&](size_t i) {
      ANN::algo::search_control ctrl{};
      //ctrl.log_per_stat = i;
      // ctrl.beta = beta;
      // ctrl.filtered = filtered;
      // ctrl.searching = true;
      res[i] = g.template search<seq_result>(q[i].get_coord(), k, ef, F[i], ctrl);
    });
  };

  // puts("Warmup");
  search();

  parlay::internal::timer t;
  // const uint32_t rounds = 3;
  for (uint32_t i = 0; i < rounds; ++i) search();
  const double time_query = t.next_time() / rounds;
  const double qps = cnt_query / time_query;
  printf("Find neighbors: %.4f s, %e kqps\n", time_query, qps / 1000);

  // printf("# visited: %lu\n", parlay::reduce(per_visited, parlay::addm<size_t>{}));
  // printf("# eval: %lu\n", parlay::reduce(per_eval, parlay::addm<size_t>{}));
  // printf("size of C: %lu\n", parlay::reduce(per_size_C, parlay::addm<size_t>{}));

  // parlay::sort_inplace(per_visited);
  // parlay::sort_inplace(per_eval);
  // parlay::sort_inplace(per_size_C);
  // const double tail_ratio[] = {0.9, 0.99, 0.999};
  // for (size_t i = 0; i < sizeof(tail_ratio) / sizeof(*tail_ratio); ++i) {
  //   const auto r = tail_ratio[i];
  //   const uint32_t tail_index = r * cnt_query;
  //   printf("%.4f tail stat (at %u):\n", r, tail_index);

  //   printf("\t# visited: %lu\n", per_visited[tail_index]);
  //   printf("\t# eval: %lu\n", per_eval[tail_index]);
  //   printf("\tsize of C: %lu\n", per_size_C[tail_index]);
  // }

  return res;
}

// For post processing
template<class G, class Seq, typename L>
auto post_processing(const G &g, const Seq &q, uint32_t k, uint32_t ef,
                     const parlay::sequence<parlay::sequence<L>> &F_b,
                     const parlay::sequence<parlay::sequence<L>> &F_q) {
  const size_t cnt_query = q.size();
  using seq_result = parlay::sequence<typename G::result_t>;
  parlay::sequence<seq_result> res(cnt_query);

  auto search = [&] {
    parlay::parallel_for(0, cnt_query, [&](size_t i) {
      ANN::algo::search_control ctrl{};
      //ctrl.log_per_stat = i;
      // ctrl.beta = beta;
      auto ret = g.template search<seq_result>(q[i].get_coord(), k, ef, ctrl);
      decltype(ret) valid;
      for (const auto &[dist, pid] : ret) {
        assert(pid < F_b.size());
        parlay::sequence<L> inter;

        std::set_intersection(F_b.begin(), F_b.end(), 
                              F_q.begin(), F_q.end(), std::back_inserter(inter));
        if (inter.size() > 0) {
          valid.push_back({dist, pid});
        }
      }
      res[i] = std::move(valid);
    });
  };

  // puts("Warmup");
  search();

  parlay::internal::timer t;
  const uint32_t rounds = 5;

  for (uint32_t i = 0; i < rounds; ++i) search();

  const double time_query = t.next_time() / rounds;
  const double qps = cnt_query / time_query;
  printf("Find neighbors: %.4f s, %e kqps\n", time_query, qps / 1000);

  return res;
}

template<typename T>
auto cand_to_gt(std::priority_queue<std::pair<float,T>> top_candidates)
{
  parlay::sequence<T> knn;
  while (!top_candidates.empty()) {
    knn.emplace_back(top_candidates.top().second);
    top_candidates.pop();
  }
  std::reverse(knn.begin(), knn.end());
  return knn;
}

template<class U, class Seq, class Point>
void _CalculateOneKnn_impl(
  const Seq &data, const Point &q, uint32_t dim, uint32_t k,
  std::priority_queue<std::pair<float, typename U::point_t::id_t>> &top_candidates)
{
  static_assert(std::is_same_v<Point, typename U::point_t>);

  float upper_bound = std::numeric_limits<float>::max();
  for (size_t i = 0; i < data.size(); ++i) {
    const auto &u = data[i];
    float dist = U::distance(u.get_coord(), q.get_coord(), dim);

    // only keep the top k
    if (top_candidates.size() < k || dist < upper_bound) {
      top_candidates.emplace(dist, u.get_id());
      if (top_candidates.size() > k) top_candidates.pop();
      upper_bound = top_candidates.top().first;
    }
  }
}

template<class U, class Seq, class Point>
auto CalculateOneKnn(const Seq &data, const Point &q, uint32_t dim, uint32_t k)
{
  using pid_t = typename U::point_t::id_t;
  std::priority_queue<std::pair<float, pid_t>> top_candidates;
  _CalculateOneKnn_impl<U>(data, q, dim, k, top_candidates);
  return cand_to_gt(std::move(top_candidates));
}

// For filter
template<class U, class Seq, class Point, typename L>
auto CalculateOneKnn(const Seq &data, const Point &q, uint32_t dim, uint32_t k,
                     const parlay::sequence<parlay::sequence<L>> &base_labels,
                     const parlay::sequence<L> &query_label) {
  static_assert(std::is_same_v<Point, typename U::point_t>);
  assert(data.size() == base_labels.size());

  using pid_t = typename U::point_t::id_t;
  std::priority_queue<std::pair<float, pid_t>> top_candidates;
  float upper_bound = std::numeric_limits<float>::max();

  for (size_t i = 0; i < data.size(); ++i) {
    const auto &u = data[i];  // id, point
    parlay::sequence<L> inter;

    std::set_intersection(base_labels[i].begin(), base_labels[i].end(),
                          query_label.begin(), query_label.end(), std::back_inserter(inter));

    if (inter.size() <= 0) continue;
    // float dist = U::distance(u.second.get_coord(), q.get_coord(), dim);
    float dist = U::distance(u.get_coord(), q.get_coord(), dim);

    // only keep the top k
    if (top_candidates.size() < k || dist < upper_bound) {
      // top_candidates.emplace(dist, u);
      top_candidates.emplace(dist, u.get_id());
      if (top_candidates.size() > k) {
        top_candidates.pop();
      }
      upper_bound = top_candidates.top().first;
    }
  }

  parlay::sequence<pid_t> knn;
  while (!top_candidates.empty()) {
    knn.emplace_back(top_candidates.top().second);
    top_candidates.pop();
  }
  std::reverse(knn.begin(), knn.end());
  return knn;
}

template<class U, class S1, class S2>
auto ConstructKnng(const S1 &data, const S2 &qs, uint32_t dim, uint32_t k) {
  using pid_t = typename U::point_t::id_t;
  parlay::sequence<parlay::sequence<pid_t>> res(qs.size());
  parlay::parallel_for(0, qs.size(), [&](size_t i) {
    res[i] = CalculateOneKnn<U>(data, qs[i], dim, k);
  });
  return res;
}

template<class U, class S1, class S2, class C>
auto ConstructKnng(
  const S1 &data, const S2 &qs, uint32_t dim, uint32_t k, C &cands)
{
  using pid_t = typename U::point_t::id_t;
  size_t qsize = qs.size();

  if(cands.size()<qs.size())
    cands.resize(qs.size());
  parlay::sequence<parlay::sequence<pid_t>> res(qs.size());
  parlay::parallel_for(0, qsize, [&](size_t i) {
    _CalculateOneKnn_impl<U>(data, qs[i], dim, k, cands[i]);
    res[i] = cand_to_gt(cands[i]);
  });
  return res;
}

// For filter
template<class U, class S1, class S2, typename L>
auto ConstructKnng(const S1 &data, const S2 &qs, uint32_t dim, uint32_t k,
                   const parlay::sequence<parlay::sequence<L>> &base_labels,
                   const parlay::sequence<parlay::sequence<L>> &query_labels) {
  using pid_t = typename U::point_t::id_t;
  parlay::sequence<parlay::sequence<pid_t>> res(qs.size());
  parlay::parallel_for(0, qs.size(), [&](size_t i) {
    res[i] = CalculateOneKnn<U>(data, qs[i], dim, k, base_labels, query_labels[i]);
  });
  return res;
}

template<class S1, class S2, class S3>
void calc_recall(const S1 &q, const S2 &res, const S3 &gt, uint32_t k) {
  const size_t cnt_query = q.size();
  //	uint32_t cnt_all_shot = 0;
  std::vector<uint32_t> result(k + 1);
  printf("measure recall@%u on %lu queries\n", k, cnt_query);
  for (uint32_t i = 0; i < cnt_query; ++i) {
    uint32_t cnt_shot = 0;
    for (uint32_t j = 0; j < k; ++j) {
      if (std::find_if(res[i].begin(), res[i].end(), [&](const auto &p) {
            // return p.dist != std::numeric_limits<decltype(p.dist)>::max() && p.pid == gt[i][j];
            return p.pid == gt[i][j];
          }) != res[i].end()) {
        cnt_shot++;
      }
    }
    result[cnt_shot]++;
  }
  size_t total_shot = 0;
  for (size_t i = 0; i <= k; ++i) {
    printf("%u ", result[i]);
    total_shot += result[i] * i;
  }
  putchar('\n');
  printf("recall: %.6f\n", float(total_shot) / cnt_query / k);
}

template<class Desc, class S1, class S2, class S3, class S4>
void calc_recall_tie(const S1 &q, const S2 &res, const S3 &gt, uint32_t k, const S4 &ps, uint32_t dim) {
  auto dist = [&](size_t gid, const auto &p){
    return Desc::distance(ps[gid].get_coord(), p.get_coord(), dim);
  };

  const size_t cnt_query = q.size();
  //  uint32_t cnt_all_shot = 0;
  std::vector<uint32_t> result(k + 1);
  printf("measure recall@%u on %lu queries\n", k, cnt_query);
  for (uint32_t i = 0; i < cnt_query; ++i) {
    size_t end = k - 1;
    while(end<gt[i].size() && dist(gt[i][end],q[i])==dist(gt[i][k-1],q[i]))
      end++;

    uint32_t cnt_shot = 0;
    for (uint32_t j = 0; j < end; ++j) {
      if (std::find_if(res[i].begin(), res[i].end(), [&](const auto &p) {
            return p.pid == gt[i][j];
          }) != res[i].end()) {
        cnt_shot++;
      }
    }

    result[cnt_shot]++;
  }
  size_t total_shot = 0;
  for (size_t i = 0; i <= k; ++i) {
    printf("%u ", result[i]);
    total_shot += result[i] * i;
  }
  putchar('\n');
  printf("recall: %.6f\n", float(total_shot) / cnt_query / k);
}


template<typename L>
inline auto load_label_helper(auto type, const char *file_label_in, size_t size_max) {
  using pid_t = decltype(type);

  parlay::internal::timer t;
  auto [F_b, P_b_v] = load_label<L, pid_t>(file_label_in, size_max, false);
  auto P_b = std::get<std::unordered_map<L, parlay::sequence<pid_t>>>(P_b_v);
  t.next("Load base labels");
  printf("Load %lu base points w/ labels\n", F_b.size());

  // auto [F_q, P_q_v] = load_label<L, pid_t>(file_label_query);
  // auto P_q = std::get<parlay::sequence<std::pair<L, parlay::sequence<pid_t>>>>(P_q_v);
  // t.next("Load query labels");
  // printf("Load %lu query points w/ labels\n\n", F_q.size());

  // return std::make_tuple(F_b, P_b, F_q, P_q);
  return std::make_tuple(F_b, P_b);
}

template<class Map, typename Nid = uint32_t, typename Label = int32_t>
auto find_medoid(const Map &P_b, const size_t n_b, float tau = 0.5) {
  // std::unordered_map<Nid, uint32_t> T;
  std::unordered_map<Label, Nid> M; // M mapping filters to start nodes

  parlay::sequence<uint32_t> T(n_b, 0); // T is intended as a counter
  // parlay::sequence<typename decltype(P_b)::mapped_type> P(P_b.size());
  // std::transform(P_b.begin(), P_b.end(), P.begin(), [](const auto& pair) {
  //   return pair.second;
  // });

  // parlay::sequence<std::pair<typename decltype(P_b)::key_type, typename decltype(P_b)::mapped_type>> P(P_b.size());
  parlay::sequence<std::pair<
    typename std::remove_reference_t<decltype(P_b)>::key_type, 
    typename std::remove_reference_t<decltype(P_b)>::mapped_type
  >> P(P_b.size());
  std::transform(P_b.begin(), P_b.end(), P.begin(), [](const auto& pair) {
    return pair;
  });

  auto rng = std::default_random_engine{};
  parlay::sequence<Nid> eps(P.size());

  parlay::parallel_for(0, P.size(), [&](size_t i) {
    auto Pf = P[i].second;  // vector of label
    if ((size_t)(Pf.size() * tau) <= 1) {
      assert((size_t)Pf[0] < n_b);
      eps[i] = Pf[0];
      T[Pf[0]] += 1;    //! test: non-atomic operation
    } else {
      // auto Pf = nids;
      std::shuffle(Pf.begin(), Pf.end(), rng);
      const size_t m = Pf.size();
      const size_t rdm = (size_t)(m * tau);
      // auto choices = parlay::sequence<decltype(Pf)::value_type>(Pf.begin(), Pf.begin() + rdm);
      auto choices = ANN::util::to<decltype(Pf)>(std::ranges::subrange(Pf.begin(), Pf.begin() + rdm));
      Nid midx{};         // fewer chose time node nid
      uint32_t mn = -1;   // node chose time
      for (auto nid : choices) {
        assert((size_t)nid < n_b);
        if (T[nid] < mn) {
          midx = nid;
          mn = T[nid];
        }
        T[nid] += 1;   //! test: non-atomic operation
      }
      eps[i] = midx;
    }
  });

  for (size_t i = 0; i < P.size(); ++i) {
    M[P[i].first] = eps[i];
  }

  return std::make_pair(eps, M);
}

// Visit all the vectors in the given 2D array of points
// This triggers the page fetching if the vectors are mmap-ed
template<class T>
void visit_point(const T &array, size_t dim0, size_t dim1)
{
  parlay::parallel_for(0, dim0, [&](size_t i){
    const auto &a = array[i];
    [[maybe_unused]] volatile auto elem = a.get_coord()[0];
    for(size_t j=1; j<dim1; ++j)
      elem = a.get_coord()[j];
  });
}

template<typename F>
auto parse_array(const std::string &s, F f){
  std::stringstream ss;
  ss << s;
  std::string current;
  std::vector<decltype(f((char*)NULL))> res;
  while(std::getline(ss, current, ','))
    res.push_back(f(current.c_str()));
  std::sort(res.begin(), res.end());
  return res;
};


template<typename T>
void write_ibin(const std::string &s, const T &data)
{
  ANN::io::buffered_writer w(s);

  w(uint32_t(data.size()));
  w(uint32_t(data.front().size()));
  w(data);
}
