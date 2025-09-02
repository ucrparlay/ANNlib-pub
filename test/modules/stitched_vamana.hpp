#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <functional>
#include <algorithm>
#include <iterator>
#include <unordered_map>
#include <type_traits>

#include "../parlay.hpp"
#include "../utils.hpp"
#include "algo/algo.hpp"
#include "algo/vamana_stitched.hpp"
#include "graph/adj.hpp"
#include "util/debug.hpp"
#include "util/intrin.hpp"

using ANN::stitched_vamana;
using ANN::util::debug_output;

template<class U, class S1, class S2, class S3>
auto run_stitched_vamana_insert(uint32_t dim, uint32_t R, uint32_t stitched_R, uint32_t efc, float alpha,
                                float batch_base, size_t size_max, const S1 &ps, const S2 &F_b,
                                const S3 &P_b) {
  using nid_t = typename stitched_vamana<U>::nid_t;
  // using pid_t = typename stitched_vamana<U>::pid_t;
  using label_t = typename stitched_vamana<U>::label_t;
  // using seq_edge = typename stitched_vamana<U>::seq_edge;
  // using seq_conn = typename stitched_vamana<U>::seq_conn;
  // using prune_control = typename stitched_vamana<U>::prune_control;
  // using cm = typename stitched_vamana<U>::cm;
  // constexpr bool filtered = false;
  (void)size_max;
/*
  auto Merge = [alpha, size_max](stitched_vamana<U> from, stitched_vamana<U> &to) -> void {

    const auto mapping = from.id_map.get_map();

    std::vector<std::pair<nid_t, pid_t>> existed_nodes, missing_nodes;

    for (const auto &node : mapping) {
      if (!to.id_map.contain_nid(node.first)) {
        missing_nodes.push_back(node);
      } else {
        existed_nodes.push_back(node);
      }
    }

    size_t n = missing_nodes.size();
    size_t m = existed_nodes.size();

    debug_output("Missing node num: %lu, Existed node num: %lu\n", n, m);

    typename stitched_vamana<U>::seq<std::pair<nid_t, seq_edge>> nbh_missing(n), nbh_existed(m);

    for (const auto &[nid, pid] : missing_nodes) {
      assert(pid < size_max);
      const auto &node = from.g.get_node(nid);
      const auto &labels = node->get_label();
      const auto &coord = node->get_coord();
      to.insert(std::make_pair(pid, nid), coord, labels);
    }

    // parallelism
    cm::parallel_for(0, n, [&](size_t i) {
      auto [nid, pid] = missing_nodes[i];
      assert(pid < size_max && to.id_map.contain_nid(nid));
      auto edges = from.g.get_edges(nid);
      // no need to prune
      nbh_missing[i] = std::make_pair(nid, edges);
    });
    to.g.set_edges(std::move(nbh_missing));

    cm::parallel_for(0, m, [&](size_t i) {
      auto [nid, pid] = existed_nodes[i];
      assert(pid < size_max);
      auto to_edges = to.g.get_edges(nid);
      const auto &from_edges = from.g.get_edges(nid);  // keep immutable here
      auto edge_v = ANN::util::to<seq_edge>(std::move(to_edges));
      edge_v.insert(edge_v.end(), std::make_move_iterator(from_edges.begin()),
                    std::make_move_iterator(from_edges.end()));
      prune_control pctrl;
      pctrl.alpha = alpha;
      seq_conn conn_v =
          // ANN::algo::prune_simple(to.conn_cast(std::move(edge_v)), to.get_deg_bound());
          ANN::algo::prune_heuristic(to.conn_cast(std::move(edge_v)), to.get_deg_bound(),
                                     to.gen_f_nbhs(), to.gen_f_dist(nid), pctrl);
      nbh_existed[i] = std::make_pair(nid, to.edge_cast(std::move(conn_v)));
    });
    to.g.set_edges(std::move(nbh_existed));
  };
*/
  parlay::sequence<std::pair<label_t, parlay::sequence<nid_t>>> Pb(P_b.begin(), P_b.end());
  /*
  sort(Pb.begin(), Pb.end(),
       [&](const std::pair<label_t, parlay::sequence<nid_t>> &a,
           const std::pair<label_t, parlay::sequence<nid_t>> &b) {
         return a.second.size() > b.second.size();
       });*/
  puts("Initialize Stitched Vamana");

  struct mixer{
    const S1 &ps;
    const S2 &fl;
    size_t i;

    decltype(auto) get_id() const{
        return ps[i].get_id();
    }
    decltype(auto) get_coord() const{
        return ps[i].get_coord();
    }
    decltype(auto) get_label() const{
        return fl[i];
    }
  };
  // puts("Insert points");
  parlay::internal::timer t;

  auto sizes = parlay::tabulate(Pb.size(), [&](size_t i){
    return Pb[i].second.size();
  });
  auto [sum,total] = parlay::scan(sizes);
  sum.push_back(total);

  std::function<stitched_vamana<U>(size_t,size_t)> merge_between = [&](size_t l, size_t r){
    if(r-l==1 || sum[r]-sum[l]<1000)
    {
      // stitched_vamana<U> base(dim, stitched_R, efc, alpha);
      stitched_vamana<U> g1(dim, R, efc, alpha);

      for(size_t i=l; i<r; ++i)
      {
        const auto &[f, Pf] = Pb[i];
        size_t n = Pf.size();
        // debug_output("Number of points w/ label [%u]: %lu\n", f, n);
        S1 new_ps(n);
        parlay::sequence<parlay::sequence<label_t>> base_labels(n);

        parlay::parallel_for(0, n, [&](size_t i) {
          auto offset = Pf[i];
          new_ps[i] = *(ps.begin() + offset);
          base_labels[i] = F_b[offset];
        });

        auto psx = ANN::util::delayed_seq(new_ps.size(), [&](size_t i){
          return mixer{new_ps, base_labels, i};
        });

        stitched_vamana<U> g2(dim, R, efc, alpha);
        g2.insert(psx.begin(), psx.end(), f, batch_base);

        if(r-l==1) return g2;

        if(g1.num_edges()>g2.num_edges())
          g1.merge(std::move(g2), false);
        else
        {
          g2.merge(std::move(g1), false);
          g1 = std::move(g2);
        }
      }
      return g1;
    }
    size_t mid = (l+r)/2;
    std::optional<stitched_vamana<U>> vamana_l, vamana_r;
    parlay::par_do(
      [&]{vamana_l = merge_between(l, mid);},
      [&]{vamana_r = merge_between(mid, r);}
    );
    if (vamana_l->num_edges() > vamana_r->num_edges()) {
      vamana_l->merge(std::move(*vamana_r), false);
      return *vamana_l;
    }
    vamana_r->merge(std::move(*vamana_l), false);
    return *vamana_r;
  };
  auto g_full = merge_between(0, Pb.size());
  t.next("Finish construction");

  puts("Collect statistics (merge only)");
  print_stat(g_full);

  g_full.prune_edges(stitched_R);
  t.next("Finish pruning");

  puts("Collect statistics (pruned)");
  print_stat(g_full);

  return g_full;
}

template<class G, class S1, class S2, class GT>
void run_stitched_vamana_search(G g, uint32_t k, uint32_t ef, const S1 &q, const S2 &F_q, const GT &gt, uint32_t rounds = 5) {
  // constexpr bool filtered = false;
  parlay::internal::timer t;

  puts("Search for neighbors");
  // g.entrance.clear();
  // auto medoid = g.template find_medoid(P_b, 0.2);
  // g.entrance =
  //     ANN::util::to<decltype(g.entrance)>(std::ranges::subrange(medoid.begin(), medoid.end()));
  auto res = find_nbhs(g, q, k, ef, F_q, rounds /*, filtered*/);
  t.next("Finish searching");

  puts("Compute recall");
  calc_recall(q, res, gt, k);

  puts("--------------------------------\n");
}
