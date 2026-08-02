#ifndef _ANN_ALGO_STITCHED_VAMANA_HPP
#define _ANN_ALGO_STITCHED_VAMANA_HPP

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iterator>
#include <utility>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <optional>
#include <parlay/parallel.h>
#include <parlay/primitives.h>

#include "algo/algo.hpp"
#include "algo/vamana.hpp"
#include "custom/custom.hpp"
#include "map/direct.hpp"
#include "map/direct_remappable.hpp"
#include "util/debug.hpp"
#include "util/helper.hpp"
#include "util/hash_map.hpp"
#include "util/small_seq.hpp"

namespace ANN::vamana_details {

// 4-byte edge: stores only the neighbor id. Distances are not persisted (search
// recomputes them; the build recomputes them at prune sites via to_conns()).
template<typename Nid>
struct stitched_edge {
  Nid u;
  constexpr bool operator<(const stitched_edge &rhs) const { return u < rhs.u; }
  constexpr bool operator>(const stitched_edge &rhs) const { return u > rhs.u; }
  constexpr bool operator==(const stitched_edge &rhs) const { return u == rhs.u; }
  constexpr bool operator!=(const stitched_edge &rhs) const { return u != rhs.u; }
};

}  // namespace ANN::vamana_details

template<typename Nid>
struct std::hash<ANN::vamana_details::stitched_edge<Nid>> {
  size_t operator()(const ANN::vamana_details::stitched_edge<Nid> &e) const noexcept {
    return std::hash<decltype(e.u)>{}(e.u);
  }
};

namespace ANN {

template<class Desc>
class stitched_vamana : public vamana<Desc> {
 public:
  using vamana<Desc>::vamana;
  using cm = typename vamana<Desc>::cm;

  using nid_t = typename vamana<Desc>::nid_t;
  using pid_t = typename vamana<Desc>::pid_t;
  using coord_t = typename vamana<Desc>::coord_t;
  using point_t = typename vamana<Desc>::point_t;
  using label_t = typename point_t::label_t;

  // using md_t = typename vamana<Desc>::md_t;
  using dist_t = typename vamana<Desc>::dist_t;
  using conn = typename vamana<Desc>::conn;
  using edge = vamana_details::stitched_edge<nid_t>;
  using search_control = typename vamana<Desc>::search_control;
  using prune_control = typename vamana<Desc>::prune_control;
  using result_t = typename vamana<Desc>::result_t;

  template<typename T>
  using seq = typename cm::seq<T>;
  using seq_edge = seq<edge>;
  using seq_conn = seq<conn>;
  using seq_label = seq<label_t>;
  using node_label_t = util::small_seq<label_t, 2>;

 private:
  struct node_t {
    coord_t coord;
    node_label_t labels;

    coord_t& get_coord() {
      return coord;
    }
    const coord_t& get_coord() const {
      return coord;
    }
    const node_label_t& get_label() const {
      return labels;
    }
  };

  using graph_t = typename Desc::graph_aux<nid_t, node_t, edge>;

 public:
  stitched_vamana(uint32_t dim, uint32_t R, uint32_t L, float alpha)
      : vamana<Desc>(dim, R, L, alpha), dim(dim), R(R), L(L), alpha(alpha) {}

 public:
  template<typename Iter>
  void insert(Iter begin, Iter end, label_t label, float batch_base = 2);

  template<typename Iter, typename EpsFn>
  void insert_with_custom_eps(Iter begin, Iter end, label_t label, const EpsFn &get_eps,
                              float batch_base = 2);

  // template<typename Iter, class E>
  // void insert(Iter begin, Iter end, const std::vector<seq_label> &F,
  //             const E& medoid, float batch_base = 2/*, bool filtered = false*/);

  template<class Seq = seq<result_t>>
  Seq search(const coord_t &cq, uint32_t k, uint32_t ef, const seq_label &F,
             const search_control &ctrl = {}) const;

  void merge(const stitched_vamana &other, bool prune=true){
    merge_impl(other, prune);
  }
  void merge(stitched_vamana &&other, bool prune=true){
    merge_impl(std::move(other), prune);
  }

  // Shift all node IDs (and edge targets) by offset.
  // After this call, every nid in g, id_map, and entrance is increased by offset.
  void remap(nid_t offset);

  void prune_edges();
  void prune_edges(uint32_t stitched_R);

  // Read-only beam search for a point not yet in the graph.
  // Returns the pruned neighbor list in this graph's nid space.
  seq_conn search_neighbors_external(const coord_t &q, label_t label) const;

  // Copy all nodes+edges from other into this graph, shifting nids by offset.
  // Equivalent to other.remap(offset) + this->merge(other) but without extra copies.
  void add_nodes_from(const stitched_vamana &other, nid_t offset);

  // Stitch precomputed cross-edges into this graph.
  //   new_nids      - nids (in this graph) of the nodes whose cross-edges are added
  //   forward_nbhs  - pruned neighbor list for new_nids[i], in source-graph nid space
  //   nbh_offset    - added to every neighbor nid to get the merged-graph nid
  void apply_cross_edges(const seq<nid_t> &new_nids,
                         const seq<seq_conn> &forward_nbhs,
                         nid_t nbh_offset);

  // NNDescent-style refinement: re-run beam search from current neighbors of
  // each nid to find better neighbors.  Improves quality after stitching.
  void refine_nodes(const seq<nid_t> &nids, uint32_t refine_L = 0);

  // Compact the underlying graph's edge array to actual max degree (if supported).
  void compact_graph() {
    if constexpr (requires { g.compact(); })
      g.compact();
  }

  // Materialize an overlay graph to a standalone flat array (if supported).
  // Must be called after stitching so parent-level overlays can use this as backing.
  void materialize_graph() {
    if constexpr (requires { g.materialize(); })
      g.materialize();
  }

  // Merge id_map and entrance from other (with offset) without copying graph data.
  // Use this when the graph is already initialized as an overlay that borrows
  // the other graph's edge data.
  void merge_meta_from(const stitched_vamana &other, nid_t offset) {
    id_map.merge_with_offset(other.id_map, offset);
    for (const auto &[lbl, nid] : other.entrance)
      entrance[lbl] = nid + offset;
  }

 public:
  // stitched_edge stores only the neighbor id (4B), so this is a real projection
  // (conn{d,u} -> edge{u}), not a reinterpret. Distances are dropped on store.
  static seq_edge edge_cast(const seq_conn &cs) {
    seq_edge out(cs.size());
    for (size_t i = 0; i < cs.size(); ++i)
      out[i] = edge{cs[i].u};
    return out;
  }

  // Rebuild (distance, id) conns for a vertex's stored neighbor edges by
  // recomputing distances from src. Needed at build-time prune sites, which
  // require distances (prune_simple/prune_heuristic operate on seq<conn>).
  seq_conn to_conns(nid_t src, const seq_edge &es) const {
    const coord_t &c = g.get_node(src)->get_coord();
    seq_conn out(es.size());
    for (size_t i = 0; i < es.size(); ++i) {
      const nid_t u = es[i].u;
      out[i] = conn{Desc::distance(c, g.get_node(u)->get_coord(), dim), u};
    }
    return out;
  }

  template<typename T>
  static seq_label get_specificity(const T &P) {
    size_t n = P.size();
    seq_label ret;

    if constexpr (std::is_same_v<T, std::unordered_map<typename T::key_type, std::vector<pid_t>>>) {
      size_t i = 1;
      for (const auto &[f, _] : P) {
        if (i == n || i == (size_t)(n * 0.75) || i == (size_t)(n * 0.5) ||
            i == (size_t)(n * 0.25) || i == std::max<size_t>(1, (size_t)(n * 0.01))) {
          ret.push_back(f);
        }
        ++i;
      }
    } else if constexpr (std::is_same_v<T, std::vector<std::pair<typename T::value_type::first_type,
                                                                 std::vector<pid_t>>>>) {
      for (size_t i = 0; i <= 100; i += 25) {
        size_t j = (i == 0 ? std::max<size_t>(1, (size_t)(n * 0.01)) : (size_t)(i / 100 * n));
        ret.push_back(P[j].first);
      }
    }

    assert(ret.size() == 5);
    return ret;
  }

 public:
  graph_t g;
  std::unordered_map<label_t, nid_t> entrance;  // To init
  map::direct_remappable<pid_t, nid_t> id_map;

 private:
  uint32_t dim;
  uint32_t R;
  uint32_t L;
  float alpha;

  // template<typename Iter>
  // void insert_batch_impl(Iter begin, Iter end);

  template<typename Iter>
  void insert_batch_impl(Iter begin, Iter end, label_t label);

  template<typename Iter, typename EpsFn>
  void insert_batch_impl_with_eps(Iter begin, Iter end, label_t label, const EpsFn &get_eps);

  template<class G>
  void merge_impl(G &&other, bool prune);

 public:
  uint32_t get_deg_bound() const {
    return R;
  }

  auto gen_f_dist(
    std::optional<std::reference_wrapper<const graph_t>> g_sec,
    std::optional<std::reference_wrapper<const coord_t>> c
  ) const
  {
    class dist_evaluator {
      std::reference_wrapper<const stitched_vamana> idx;
      std::optional<std::reference_wrapper<const graph_t>> g_sec;
      std::optional<std::reference_wrapper<const coord_t>> c;
      uint32_t dim;

     public:
      dist_evaluator(
        std::reference_wrapper<const stitched_vamana> idx, 
        std::optional<std::reference_wrapper<const graph_t>> g_sec,
        std::optional<std::reference_wrapper<const coord_t>> c,
        uint32_t dim) :
        idx(idx), g_sec(g_sec), c(c), dim(dim)
      {
      }

      dist_t operator()(nid_t v) const{
        return Desc::distance((*c).get(), idx.get().g.get_node(v)->get_coord(), dim);
      }
      dist_t operator()(nid_t u, nid_t v) const{
        auto get_coord = [&](nid_t u) -> decltype(auto){
          return !g_sec||idx.get().id_map.contain_nid(u)?
            idx.get().g.get_node(u)->get_coord():
            (*g_sec).get().get_node(u)->get_coord();
        };
        const coord_t &cu = get_coord(u);
        const coord_t &cv = get_coord(v);
        return Desc::distance(cu, cv, dim);
      }
    };

    return dist_evaluator(*this, g_sec, c, dim);
  }

  auto gen_f_dist(const graph_t &g_sec) const{
    return gen_f_dist(std::cref(g_sec), std::nullopt);
  }
  auto gen_f_dist(const coord_t &c) const{
    return gen_f_dist(std::nullopt, std::cref(c));
  }
  auto gen_f_dist(nid_t u) const {
    return gen_f_dist(g.get_node(u)->get_coord());
  }
  auto gen_f_dist() const{
    return gen_f_dist(std::nullopt, std::nullopt);
  }

  template<class LSeq>
  auto gen_f_nbhs(const LSeq &lb_f) const{
    // return [&,htbl=std::unordered_set<label_t>(lb_f.begin(),lb_f.end())](nid_t u){
    return [&](nid_t u){
      auto fl = std::views::filter([&](const edge &e){
        const auto &lb_u = g.get_node(e.u)->get_label();
      
        auto first1 = std::ranges::begin(lb_u);
        auto last1 = std::ranges::end(lb_u);
        auto first2 = std::ranges::begin(lb_f);
        auto last2 = std::ranges::end(lb_f);
        while(first1!=last1 && first2!=last2)
        {
          if (*first1<*first2)
            ++first1;
          else
          {
            if (!(*first2<*first1))
              return true;
            ++first2;
          }
        }
        return false;
        /*
        for(label_t l : lb_u)
          if(htbl.contains(l))
            return true;
        return false;*/
      });

      auto t = std::views::transform([&](const edge &e) { return e.u; });

      if constexpr(std::is_reference_v<decltype(g.get_edges(u))>)
        return std::ranges::ref_view(g.get_edges(u)) | fl | t;
      else
        return std::ranges::owning_view(g.get_edges(u)) | fl | t;
    };
  }
  auto gen_f_nbhs(nid_t p) const{
    return gen_f_nbhs(g.get_node(p)->get_label());
  }
  auto gen_f_nbhs() const {
    return [&](nid_t u) {
      // auto f = std::views::filter([&](const edge &e) {
      //   auto &ls = e.livestamp;
      //   if (ls == 0) return false;
      //   if (ls == deltick) return true;
      //   ls = id_map.contain_nid(e.u) ? deltick : 0;
      //   return ls != 0;
      // });

      auto t = std::views::transform([&](const edge &e) {
        return e.u;
      });

      if constexpr (std::is_reference_v<decltype(g.get_edges(u))>) {
        return std::ranges::ref_view(g.get_edges(u)) /*| f */ | t;
      } else {
        return std::ranges::owning_view(g.get_edges(u)) /*| f */ | t;
      }
    };
  }

  auto gen_f_label(nid_t u) const{
    class label_extractor{
      std::reference_wrapper<const graph_t> g;
      nid_t u;
    public:
      label_extractor(const graph_t &g, nid_t u): g(g), u(u){}
      decltype(auto) operator()() const{
        return g.get().get_node(u)->get_label();
      }
      decltype(auto) operator()(nid_t v) const{
        return g.get().get_node(v)->get_label();
      }
    };

    return label_extractor(g, u);
  }

  template<class Op>
  auto calc_degs(Op op) const {
    seq<size_t> degs(cm::num_workers(), 0);
    g.for_each([&](auto p) {
      auto &deg = degs[cm::worker_id()];
      deg = op(deg, num_edges(p.get_id()));
    });
    return cm::reduce(degs, size_t(0), op);
  }

  size_t num_nodes() const {
    return g.num_nodes();
  }

  size_t num_edges(nid_t u) const {
    // return g.get_edges(u).size();
    return std::ranges::distance(gen_f_nbhs()(u));
  }
  size_t num_edges() const {
    return calc_degs(std::plus<>{});
  }

  size_t max_deg() const {
    return calc_degs([](size_t x, size_t y) {
      return std::max(x, y);
    });
  }
};

template<class Desc>
template<typename Iter>
void stitched_vamana<Desc>::insert(Iter begin, Iter end, label_t label, float batch_base/*, bool filtered*/) {
  // static_assert(std::is_same_v<typename std::iterator_traits<Iter>::value_type, point_t>);
  static_assert(std::is_base_of_v<std::random_access_iterator_tag,
                                  typename std::iterator_traits<Iter>::iterator_category>);

  const size_t n = std::distance(begin, end);
  if (n == 0) return;

  // std::random_device rd;
  // auto perm = cm::random_permutation(n /*, rd()*/);
  auto rand_seq = util::delayed_seq(n, [&](size_t i) -> decltype(auto) {
    return *(begin + i /*perm[i]*/);
  });
  // auto rand_label_seq =
  //     util::delayed_seq(n, [&](size_t i) -> decltype(auto) { return *(F.begin() + perm[i]); });

  size_t cnt_skip = 0;
  if (g.empty()) {
    // const nid_t ep_init = id_map.insert(rand_seq.begin()->get_id());
    auto init = rand_seq.begin();
    const nid_t ep = id_map.insert(init->get_id());
    g.add_node(ep, node_t{init->get_coord(), util::to<node_label_t>(init->get_label())});
    // const nid_t ep_init = id_map.insert(static_cast<pid_t>(it->get_id()));
    // g.add_node(ep_init, node_t{it->get_coord(), *(F.begin())});
    entrance[label] = ep;
    cnt_skip = 1;
  }

  size_t batch_begin = 0, batch_end = cnt_skip, size_limit = std::max<size_t>(n * 0.02, 20000);
  // float progress = 0.0;

  while (batch_end < n) {
    batch_begin = batch_end;
    batch_end = std::min<size_t>(
        {n, (size_t)std::ceil(batch_begin * batch_base) + 1, batch_begin + size_limit});

    // std::cerr << "(batch_begin, batch_end)" << batch_begin << " " << batch_end << '\n';

    util::debug_output("Batch insertion: [%u, %u)\n", batch_begin, batch_end);
    // insert_batch_impl(rand_seq.begin()+batch_begin, rand_seq.begin()+batch_end);
    insert_batch_impl(rand_seq.begin() + batch_begin, rand_seq.begin() + batch_end, label);
    // insert(rand_seq.begin()+batch_begin, rand_seq.begin()+batch_end, false);

    // if (batch_end > n * (progress + 0.05)) {
    //   progress = float(batch_end) / n;
    //   fprintf(stderr, "Built: %3.2f%%\n", progress * 100);
    //   fprintf(stderr, "# visited: %lu\n", cm::reduce(per_visited));
    //   fprintf(stderr, "# eval: %lu\n", cm::reduce(per_eval));
    //   fprintf(stderr, "size of C: %lu\n", cm::reduce(per_size_C));
    //   per_visited.clear();
    //   per_eval.clear();
    //   per_size_C.clear();
    // }
  }

  // fprintf(stderr, "# visited: %lu\n", cm::reduce(per_visited));
  // fprintf(stderr, "# eval: %lu\n", cm::reduce(per_eval));
  // fprintf(stderr, "size of C: %lu\n", cm::reduce(per_size_C));
  // per_visited.clear();
  // per_eval.clear();
  // per_size_C.clear();
}

template<class Desc>
template<typename Iter, typename EpsFn>
void stitched_vamana<Desc>::insert_with_custom_eps(
    Iter begin, Iter end, label_t label, const EpsFn &get_eps, float batch_base) {
  static_assert(std::is_base_of_v<std::random_access_iterator_tag,
                                  typename std::iterator_traits<Iter>::iterator_category>);

  const size_t n = std::distance(begin, end);
  if (n == 0) return;

  auto rand_seq = util::delayed_seq(n, [&](size_t i) -> decltype(auto) {
    return *(begin + i);
  });

  size_t cnt_skip = 0;
  if (g.empty()) {
    auto init = rand_seq.begin();
    const nid_t ep = id_map.insert(init->get_id());
    g.add_node(ep, node_t{init->get_coord(), util::to<node_label_t>(init->get_label())});
    entrance[label] = ep;
    cnt_skip = 1;
  }

  size_t batch_begin = 0, batch_end = cnt_skip, size_limit = std::max<size_t>(n * 0.02, 20000);
  while (batch_end < n) {
    batch_begin = batch_end;
    batch_end = std::min<size_t>(
        {n, (size_t)std::ceil(batch_begin * batch_base) + 1, batch_begin + size_limit});

    util::debug_output("Batch insertion: [%u, %u)\n", batch_begin, batch_end);
    insert_batch_impl_with_eps(
        rand_seq.begin() + batch_begin, rand_seq.begin() + batch_end, label,
        [&](size_t i, nid_t u) {
          return get_eps(batch_begin + i, u);
        });
  }
}

template<class Desc>
template<typename Iter>
void stitched_vamana<Desc>::insert_batch_impl(Iter begin, Iter end, label_t label) {
  const size_t batch_size = std::distance(begin, end);
  seq<nid_t> nids(batch_size);

  // per_visited.resize(batch_size);
  // per_eval.resize(batch_size);
  // per_size_C.resize(batch_size);

  // before the insertion, prepare the needed data
  // `nids[i]` is the nid of the node corresponding to the i-th
  // point to insert in the batch, associated with level[i]
  id_map.insert(util::delayed_seq(batch_size, [&](size_t i) {
    return (begin + i)->get_id();
  }));

  cm::parallel_for(0, batch_size, [&](uint32_t i) {
    nids[i] = id_map.get_nid((begin + i)->get_id());
  });

  g.add_nodes(util::delayed_seq(batch_size, [&](size_t i) {
    auto it = begin + i;
    // GUARANTEE: begin[*].get_coord is only invoked for assignment once
    return std::pair{nids[i], node_t{it->get_coord(), util::to<node_label_t>(it->get_label())}};
    // return std::pair{nids[i], node_t{(begin + i)->get_coord(), *(F.begin() + i)}};
  }));

  // below we (re)generate edges incident to nodes in the current batch
  // add adges from the new points
  seq<seq<std::pair<nid_t, edge>>> edge_added(batch_size);
  seq<std::pair<nid_t, seq_edge>> nbh_forward(batch_size);

  cm::parallel_for(0, batch_size, [&](size_t i) {
    const nid_t u = nids[i];

    // auto &eps_u = entrance;
    search_control sctrl;  // TODO: use designated initializers in C++20
    // sctrl.filtered = filtered;
    // sctrl.searching = false;
    seq_conn res = algo::beamSearch(
      gen_f_nbhs(nids[i]), gen_f_dist(u), seq<nid_t>{entrance.at(label)}, L, sctrl
    ).second;

    prune_control pctrl;  // TODO: use designated intializers in C++20
    pctrl.alpha = alpha;
    seq_conn conn_u =
        algo::prune_heuristic(std::move(res), get_deg_bound(), gen_f_nbhs(), gen_f_dist(u), pctrl);
    // record the edge for the backward insertion later
    auto &edge_cur = edge_added[i];
    edge_cur.clear();
    edge_cur.reserve(conn_u.size());
    for (const auto &[d, v] : conn_u) {
      edge_cur.emplace_back(v, edge{u});
    }

    // store for batch insertion
    nbh_forward[i] = {u, edge_cast(std::move(conn_u))};
  });
  util::debug_output("Adding forward edges\n");
  g.set_edges(std::move(nbh_forward));

  // now we add edges in the other direction
  auto edge_added_flatten = util::flatten(std::move(edge_added));
  auto edge_added_grouped = util::group_by_key(std::move(edge_added_flatten));

  using agent_t = std::remove_cvref_t<decltype(g.get_edges(nid_t()))>;
  seq<std::pair<nid_t, agent_t>> nbh_backward(edge_added_grouped.size());

  if constexpr (requires { g.ensure_max_degree(uint32_t{}); })
    g.ensure_max_degree(get_deg_bound());

  cm::parallel_for(0, edge_added_grouped.size(), [&](size_t j) {
    nid_t v = edge_added_grouped[j].first;
    auto &nbh_v_add = edge_added_grouped[j].second;

    auto &&edge_agent_v = g.get_edges(v);
    auto edge_v = util::to<seq_edge>(std::move(edge_agent_v));
    // auto edge_v =
    //     util::to<seq_edge>(std::move(edge_agent_v) | std::views::filter([&](const edge &e) {
    //                          return id_map.contain_nid(e.u);
    //                        }));
    edge_v.insert(edge_v.end(), std::make_move_iterator(nbh_v_add.begin()),
                  std::make_move_iterator(nbh_v_add.end()));

    seq_conn conn_v = algo::prune_simple(to_conns(v, edge_v), get_deg_bound());
    edge_agent_v = edge_cast(conn_v); // TODO: check if to use std::move
    nbh_backward[j] = {v, std::move(edge_agent_v)};
  });
  util::debug_output("Adding backward edges\n");
  g.set_edges(std::move(nbh_backward));

  // finally, update the entrances
  // entrance.insert(entrance.end(), nids.begin(), nids.end());
}

template<class Desc>
template<typename Iter, typename EpsFn>
void stitched_vamana<Desc>::insert_batch_impl_with_eps(
    Iter begin, Iter end, label_t label, const EpsFn &get_eps) {
  const size_t batch_size = std::distance(begin, end);
  seq<nid_t> nids(batch_size);

  id_map.insert(util::delayed_seq(batch_size, [&](size_t i) {
    return (begin + i)->get_id();
  }));

  cm::parallel_for(0, batch_size, [&](uint32_t i) {
    nids[i] = id_map.get_nid((begin + i)->get_id());
  });

  g.add_nodes(util::delayed_seq(batch_size, [&](size_t i) {
    auto it = begin + i;
    return std::pair{nids[i], node_t{it->get_coord(), util::to<node_label_t>(it->get_label())}};
  }));

  seq<seq<std::pair<nid_t, edge>>> edge_added(batch_size);
  seq<std::pair<nid_t, seq_edge>> nbh_forward(batch_size);

  cm::parallel_for(0, batch_size, [&](size_t i) {
    const nid_t u = nids[i];

    auto eps = get_eps(i, u);
    if (eps.empty()) {
      auto it = entrance.find(label);
      if (it != entrance.end()) {
        eps.push_back(it->second);
      }
    }
    assert(!eps.empty());

    search_control sctrl;
    seq_conn res = algo::beamSearch(gen_f_nbhs(nids[i]), gen_f_dist(u), eps, L, sctrl).second;

    prune_control pctrl;
    pctrl.alpha = alpha;
    seq_conn conn_u =
        algo::prune_heuristic(std::move(res), get_deg_bound(), gen_f_nbhs(), gen_f_dist(u), pctrl);

    auto &edge_cur = edge_added[i];
    edge_cur.clear();
    edge_cur.reserve(conn_u.size());
    for (const auto &[d, v] : conn_u) {
      edge_cur.emplace_back(v, edge{u});
    }

    nbh_forward[i] = {u, edge_cast(std::move(conn_u))};
  });
  util::debug_output("Adding forward edges\n");
  g.set_edges(std::move(nbh_forward));

  auto edge_added_flatten = util::flatten(std::move(edge_added));
  auto edge_added_grouped = util::group_by_key(std::move(edge_added_flatten));

  using agent_t = std::remove_cvref_t<decltype(g.get_edges(nid_t()))>;
  seq<std::pair<nid_t, agent_t>> nbh_backward(edge_added_grouped.size());

  if constexpr (requires { g.ensure_max_degree(uint32_t{}); })
    g.ensure_max_degree(get_deg_bound());

  cm::parallel_for(0, edge_added_grouped.size(), [&](size_t j) {
    nid_t v = edge_added_grouped[j].first;
    auto &nbh_v_add = edge_added_grouped[j].second;

    auto &&edge_agent_v = g.get_edges(v);
    auto edge_v = util::to<seq_edge>(std::move(edge_agent_v));
    edge_v.insert(edge_v.end(), std::make_move_iterator(nbh_v_add.begin()),
                  std::make_move_iterator(nbh_v_add.end()));

    seq_conn conn_v = algo::prune_simple(to_conns(v, edge_v), get_deg_bound());
    edge_agent_v = edge_cast(conn_v);
    nbh_backward[j] = {v, std::move(edge_agent_v)};
  });
  util::debug_output("Adding backward edges\n");
  g.set_edges(std::move(nbh_backward));
}

template<class Desc>
template<class Seq>
Seq stitched_vamana<Desc>::search(const coord_t &cq, uint32_t k, uint32_t ef,
                                  const seq_label &F, const search_control &ctrl) const {
  // seq<nid_t> eps = entrance;
  // auto nbhs = beamSearch(gen_f_nbhs(), gen_f_dist(cq), eps, ef, ctrl);
  auto eps = F | std::views::transform([&](label_t l) { return entrance.at(l); });
  auto nbhs = algo::beamSearch(gen_f_nbhs(F), gen_f_dist(cq), eps, ef, ctrl).first;

  cm::sort(nbhs.begin(), nbhs.end());
  nbhs = algo::prune_simple(std::move(nbhs), k /*, ctrl*/);  // TODO: set ctrl

  using result_t = typename Seq::value_type;
  static_assert(util::is_direct_list_initializable_v<result_t, dist_t, pid_t>);

  Seq res(nbhs.size());
  cm::parallel_for(0, nbhs.size(), [&](size_t i) {
    const auto &nbh = nbhs[i];
    res[i] = result_t{nbh.d, id_map.get_pid(nbh.u)};
  });

  return res;
}

template<class Desc>
void stitched_vamana<Desc>::remap(nid_t offset)
{
  if (offset == 0) return;

  // Collect valid nids from id_map (before we modify anything)
  seq<nid_t> old_nids;
  old_nids.reserve(id_map.size());
  for (nid_t nid = 0; nid < (nid_t)id_map.mapping.size(); ++nid)
    if (id_map.contain_nid(nid))
      old_nids.push_back(nid);
  const size_t n = old_nids.size();

  // Copy node Ext data and edges (with offset applied to edge targets)
  seq<node_t>   node_exts(n);
  seq<seq_edge> node_edges(n);
  cm::parallel_for(0, n, [&](size_t i) {
    nid_t old_nid = old_nids[i];
    node_exts[i] = *g.get_node(old_nid);
    auto old_e = g.get_edges(old_nid);
    seq_edge edges(old_e.begin(), old_e.end());
    for (auto &e : edges) e.u += offset;
    node_edges[i] = std::move(edges);
  });

  // Add nodes at shifted positions (resizes the adj array to offset+N)
  g.add_nodes(util::delayed_seq(n, [&](size_t i) {
    return std::pair{old_nids[i] + offset, node_exts[i]};
  }));

  // Set edges at shifted positions
  seq<std::pair<nid_t, seq_edge>> edge_pairs(n);
  cm::parallel_for(0, n, [&](size_t i) {
    edge_pairs[i] = {old_nids[i] + offset, std::move(node_edges[i])};
  });
  g.set_edges(std::move(edge_pairs));

  // Update id_map and entrance
  id_map.remap(offset);
  for (auto &[label, nid] : entrance) nid += offset;
}

template<class Desc>
template<class G>
void stitched_vamana<Desc>::merge_impl(G &&other, bool prune)
{
  // static_assert(map_t::is_pure);
  static_assert(decltype(id_map)::is_pure);
  using ea_t = std::remove_cvref_t<decltype(g.get_edges(nid_t()))>;

  // After remap(), other.g may have empty slots (adj array larger than id_map).
  // Use id_map-based iteration to process only valid nodes.
  assert(id_map.size() <= g.num_nodes() + 1);
  assert(other.id_map.size() <= other.g.num_nodes() + 1);

  // Collect valid nids from other's id_map for parallel iteration
  seq<nid_t> other_nids;
  other_nids.reserve(other.id_map.size());
  for (nid_t nid = 0; nid < (nid_t)other.id_map.mapping.size(); ++nid)
    if (other.id_map.contain_nid(nid))
      other_nids.push_back(nid);
  const size_t other_n = other_nids.size();

  util::hash_map<nid_t, node_t> nodes_new(other_n);
  cm::parallel_for(0, other_n, [&](size_t i) {
    nid_t u = other_nids[i];
    if (!id_map.contain_nid(u)) {
      auto p = other.g.get_node(u);
      nodes_new.insert(u, util::forward_like<G>(*p), true);
    }
  });
  g.add_nodes(std::move(nodes_new).kvs());

  util::hash_map<nid_t, ea_t> nbh_new(other_n);
  if constexpr (requires { g.ensure_max_degree(uint32_t{}); })
    g.ensure_max_degree(get_deg_bound() * 2);
  cm::parallel_for(0, other_n, [&](size_t i) {
    nid_t u = other_nids[i];
    if(id_map.contain_nid(u))
    {
      auto &&ea_this = g.get_edges(u);
      auto edge_this = util::to<seq_edge>(std::move(ea_this));

      auto p = other.g.get_node(u);
      auto &&ea_other = other.g.get_edges(p);
      if constexpr(std::is_rvalue_reference_v<G&&>)
        edge_this.insert(
          edge_this.end(),
          std::make_move_iterator(ea_other.begin()),
          std::make_move_iterator(ea_other.end())
        );
      else
        edge_this.insert(edge_this.end(), ea_other.begin(), ea_other.end());

      if(prune)
      {
        seq_conn conn_u = algo::prune_heuristic(
          to_conns(u, edge_this), get_deg_bound(),
          gen_f_nbhs(), gen_f_dist(), prune_control{.alpha=alpha}
        );
        ea_this = edge_cast(std::move(conn_u));
      }
      else
      {
        ea_this = std::move(edge_this);
      }
      nbh_new.insert(u, std::move(ea_this), true);
    }
    else
    {
      auto p = other.g.get_node(u);
      auto &&ea_other = other.g.get_edges(p);
      nbh_new.insert(u, util::forward_like<G>(ea_other), true);
    }
  });

  id_map.merge(util::forward_like<G>(other.id_map));
  g.set_edges(std::move(nbh_new).kvs());
  if constexpr(std::is_const_v<std::remove_reference_t<G>>)
    entrance.insert(other.entrance.begin(), other.entrance.end());
  else
    entrance.merge(std::move(other.entrance));
}

template<class Desc>
void stitched_vamana<Desc>::prune_edges()
{
  using ea_t = std::remove_cvref_t<decltype(g.get_edges(nid_t()))>;

  util::hash_map<nid_t, ea_t> nbh_new(g.num_nodes());
  g.for_each([&](auto p){
    nid_t u = p.get_id();
    // REQUIRE: other has the same type as *this
    // use auto&& to be compatible with refs
    auto &&ea_this = g.get_edges(u);
    auto edge_this = util::to<seq_edge>(std::move(ea_this));
    // seq_conn conn_u = algo::prune_heuristic(
    seq_conn conn_u = algo::prune_heuristic_filtered(
      to_conns(u, edge_this), get_deg_bound(),
      // gen_f_nbhs(), gen_f_dist(), prune_control{.alpha=alpha}
      gen_f_dist(), gen_f_label(u), prune_control{.alpha=alpha}
    );
    ea_this = edge_cast(std::move(conn_u));
    nbh_new.insert(u, std::move(ea_this), true);
  });

  g.set_edges(std::move(nbh_new).kvs());
}

template<class Desc>
void stitched_vamana<Desc>::prune_edges(uint32_t stitched_R)
{
  using ea_t = std::remove_cvref_t<decltype(g.get_edges(nid_t()))>;

  util::hash_map<nid_t, ea_t> nbh_new(g.num_nodes());
  g.for_each([&](auto p){
    nid_t u = p.get_id();
    // REQUIRE: other has the same type as *this
    // use auto&& to be compatible with refs
    auto &&ea_this = g.get_edges(u);
    auto edge_this = util::to<seq_edge>(std::move(ea_this));
    // seq_conn conn_u = algo::prune_heuristic_filtered(
    seq_conn conn_u = algo::prune_heuristic(
      // to_conns(u, edge_this), get_deg_bound(),
      // gen_f_dist(), gen_f_label(u), prune_control{.alpha=alpha}
      to_conns(u, edge_this), stitched_R,
      gen_f_nbhs(), gen_f_dist(), prune_control{.alpha=alpha}
    );
    ea_this = edge_cast(std::move(conn_u));
    nbh_new.insert(u, std::move(ea_this), true);
  });

  g.set_edges(std::move(nbh_new).kvs());
}

template<class Desc>
typename stitched_vamana<Desc>::seq_conn
stitched_vamana<Desc>::search_neighbors_external(const coord_t &q, label_t label) const {
  auto it = entrance.find(label);
  if (it == entrance.end()) return {};

  seq_label lb{label};
  seq<nid_t> eps{it->second};
  search_control sctrl;
  seq_conn res = algo::beamSearch(gen_f_nbhs(lb), gen_f_dist(q), eps, L, sctrl).second;

  prune_control pctrl;
  pctrl.alpha = alpha;
  return algo::prune_heuristic(std::move(res), get_deg_bound(), gen_f_nbhs(), gen_f_dist(q), pctrl);
}

template<class Desc>
void stitched_vamana<Desc>::add_nodes_from(const stitched_vamana &other, nid_t offset) {
  seq<nid_t> other_nids;
  other_nids.reserve(other.id_map.size());
  for (nid_t nid = 0; nid < (nid_t)other.id_map.mapping.size(); ++nid)
    if (other.id_map.contain_nid(nid))
      other_nids.push_back(nid);
  const size_t other_n = other_nids.size();

  g.add_nodes(util::delayed_seq(other_n, [&](size_t i) {
    nid_t old_nid = other_nids[i];
    return std::pair{old_nid + offset, *other.g.get_node(old_nid)};
  }));

  seq<std::pair<nid_t, seq_edge>> edge_pairs(other_n);
  cm::parallel_for(0, other_n, [&](size_t i) {
    nid_t old_nid = other_nids[i];
    auto old_edges = other.g.get_edges(old_nid);
    seq_edge edges(old_edges.begin(), old_edges.end());
    for (auto &e : edges) e.u += offset;
    edge_pairs[i] = {old_nid + offset, std::move(edges)};
  });
  g.set_edges(std::move(edge_pairs));

  id_map.merge_with_offset(other.id_map, offset);

  for (const auto &[lbl, nid] : other.entrance)
    entrance[lbl] = nid + offset;
}

template<class Desc>
void stitched_vamana<Desc>::apply_cross_edges(const seq<nid_t> &new_nids,
                                               const seq<seq_conn> &forward_nbhs,
                                               nid_t nbh_offset) {
  const size_t n = new_nids.size();
  if (n == 0) return;

  if constexpr (requires { g.ensure_max_degree(uint32_t{}); })
    g.ensure_max_degree(get_deg_bound() * 2);

  seq<seq<std::pair<nid_t, edge>>> edge_added(n);
  seq<std::pair<nid_t, seq_edge>> nbh_forward(n);

  cm::parallel_for(0, n, [&](size_t i) {
    const nid_t u = new_nids[i];
    const seq_conn &conn_u_src = forward_nbhs[i];

    auto &edge_cur = edge_added[i];
    edge_cur.reserve(conn_u_src.size());
    for (const auto &[d, v_src] : conn_u_src) {
      nid_t v = v_src + nbh_offset;
      edge_cur.emplace_back(v, edge{u});
    }

    auto edge_u = util::to<seq_edge>(g.get_edges(u));
    for (const auto &[d, v_src] : conn_u_src) {
      nid_t v = v_src + nbh_offset;
      edge_u.push_back(edge{v});
    }
    seq_conn conn_pruned = algo::prune_heuristic(
        to_conns(u, edge_u), get_deg_bound(),
        gen_f_nbhs(), gen_f_dist(u), prune_control{.alpha = alpha});
    nbh_forward[i] = {u, edge_cast(std::move(conn_pruned))};
  });
  g.set_edges(std::move(nbh_forward));

  auto edge_added_flatten = util::flatten(std::move(edge_added));
  auto edge_added_grouped = util::group_by_key(std::move(edge_added_flatten));

  using agent_t = std::remove_cvref_t<decltype(g.get_edges(nid_t()))>;
  seq<std::pair<nid_t, agent_t>> nbh_backward(edge_added_grouped.size());

  cm::parallel_for(0, edge_added_grouped.size(), [&](size_t j) {
    nid_t v = edge_added_grouped[j].first;
    auto &nbh_v_add = edge_added_grouped[j].second;

    auto &&edge_agent_v = g.get_edges(v);
    auto edge_v = util::to<seq_edge>(std::move(edge_agent_v));
    edge_v.insert(edge_v.end(), std::make_move_iterator(nbh_v_add.begin()),
                  std::make_move_iterator(nbh_v_add.end()));
    seq_conn conn_v = algo::prune_simple(to_conns(v, edge_v), get_deg_bound());
    edge_agent_v = edge_cast(conn_v);
    nbh_backward[j] = {v, std::move(edge_agent_v)};
  });
  g.set_edges(std::move(nbh_backward));
}

template<class Desc>
void stitched_vamana<Desc>::refine_nodes(const seq<nid_t> &nids, uint32_t refine_L) {
  if (nids.empty()) return;
  if (refine_L == 0) refine_L = L;

  seq<std::pair<nid_t, seq_edge>> nbh_forward(nids.size());

  cm::parallel_for(0, nids.size(), [&](size_t i) {
    const nid_t u = nids[i];
    auto cur_edges = util::to<seq_edge>(g.get_edges(u));
    if (cur_edges.empty()) { nbh_forward[i] = {u, {}}; return; }

    seq<nid_t> eps;
    eps.reserve(cur_edges.size());
    for (const auto &e : cur_edges) eps.push_back(e.u);

    search_control sctrl;
    seq_conn res = algo::beamSearch(gen_f_nbhs(), gen_f_dist(u), eps, refine_L, sctrl).second;

    prune_control pctrl;
    pctrl.alpha = alpha;
    seq_conn conn_pruned = algo::prune_heuristic(
        std::move(res), get_deg_bound(), gen_f_nbhs(), gen_f_dist(u), pctrl);
    nbh_forward[i] = {u, edge_cast(std::move(conn_pruned))};
  });

  g.set_edges(std::move(nbh_forward));
}

}  // namespace ANN

#endif  // _ANN_ALGO_STITCHED_VAMANA_HPP
