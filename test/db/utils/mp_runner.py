import time
import logging
import traceback
import concurrent

import numpy as np
import multiprocessing as mp
from typing import Iterable, List

log = logging.getLogger(__name__)


class MultiProcessingSearchRunner:
    """ multiprocessing search runner

    Args:
        k(int): search topk, default to 100
        concurrency(Iterable): concurrencies, default [1, 5, 10, 15, 20, 25, 30, 35]
        duration(int): duration for each concurency, default to 30s
    """

    def __init__(
        self,
        func,
        name: str,
        metric: str,
        data,
        labels: List[int],
        gt: List[list[int]],
        k: int = 10,
        concurrencies: Iterable[int] = [64,],
        duration: int = 30,
    ):
        self.func = func
        self.name = name
        self.metric = metric
        self.k = k
        self.concurrencies = concurrencies
        self.duration = duration

        self.data = data
        self.labels = labels
        log.debug(f"test dataset columns: {len(data)}")
        self.gt = gt
        self.ef = 100

    def search(self, data, labels: List[int], gt: List[list[int]], q: mp.Queue, cond) -> tuple[int, float]:
        """
        Worker function run in each process.

        It synchronizes with the other processes via a shared Queue and Condition,
        then repeatedly calls the search function until the duration expires.
        """
        # sync all process
        q.put(1)
        with cond:
            cond.wait()

        num, idx = len(data), 0
        assert len(labels) == num

        start_time = time.perf_counter()
        count = 0
        latencies = []
        hits = 0
        while time.perf_counter() < start_time + self.duration:
            s = time.perf_counter()
            try:
                res = self.func(self.name, self.metric,
                                data[idx], labels[idx], self.k, self.ef)
            except Exception as e:
                log.warning(f"VectorDB search_embedding error: {e}")
                traceback.print_exc(chain=True)
                raise e from None

            latencies.append(time.perf_counter() - s)
            hits += len(set(res).intersection(set(gt[idx])))
            count += 1
            # loop through the test data
            idx = idx + 1 if idx < num - 1 else 0

            if count % 500 == 0:
                print(
                    f"({mp.current_process().name:16}) search_count: {count}, latest_latency={time.perf_counter()-s}")

        total_dur = round(time.perf_counter() - start_time, 4)
        log.info(
            f"{mp.current_process().name:16} search {self.duration}s: "
            f"actual_dur={total_dur}s, count={count}, qps in this process: {round(count / total_dur, 4):3}"
        )

        return (count, total_dur, latencies, hits)

    @staticmethod
    def get_mp_context():
        mp_start_method = "spawn"
        log.debug(
            f"MultiProcessingSearchRunner get multiprocessing start method: {mp_start_method}")
        return mp.get_context(mp_start_method)

    def _run_all_concurrencies_mem_efficient(self) -> float:
        max_qps = 0
        conc_num_list = []
        conc_qps_list = []
        conc_latency_p99_list = []
        conc_recall_list = []

        try:
            for conc in self.concurrencies:
                with mp.Manager() as m:
                    q, cond = m.Queue(), m.Condition()
                    with concurrent.futures.ProcessPoolExecutor(mp_context=self.get_mp_context(), max_workers=conc) as executor:
                        print(
                            f"Start search {self.duration}s in concurrency {conc}")
                        future_iter = [executor.submit(
                            self.search, self.data, self.labels, self.gt, q, cond) for _ in range(conc)]
                        # Sync all processes
                        while q.qsize() < conc:
                            # print("qsize: ", q.qsize())
                            sleep_t = conc if conc < 10 else 10
                            time.sleep(sleep_t)

                        with cond:
                            cond.notify_all()
                            print(
                                f"Syncing all process and start concurrency search, concurrency={conc}")

                        start = time.perf_counter()
                        all_count = sum([r.result()[0] for r in future_iter])
                        latencies = sum([r.result()[2]
                                        for r in future_iter], start=[])
                        all_hits = sum([r.result()[3] for r in future_iter])
                        latency_p99 = np.percentile(latencies, 0.99)
                        cost = time.perf_counter() - start
                        # recall = (all_hits, all_count)
                        recall = 100. * all_hits / all_count / self.k

                        qps = round(all_count / cost, 4)
                        conc_num_list.append(conc)
                        conc_qps_list.append(qps)
                        conc_latency_p99_list.append(latency_p99)
                        conc_recall_list.append(recall)
                        print(
                            f"End search in concurrency {conc}: dur={cost}s, total_count={all_count}, qps={qps}")

                if qps > max_qps:
                    max_qps = qps
                    print(
                        f"Update largest qps with concurrency {conc}: current max_qps={max_qps}")
        except Exception as e:
            log.warning(
                f"Fail to search all concurrencies: {self.concurrencies}, max_qps before failure={max_qps}, reason={e}")
            traceback.print_exc()

            # No results available, raise exception
            if max_qps == 0.0:
                raise e from None

        finally:
            self.stop()

        return (max_qps, conc_num_list, conc_qps_list, conc_latency_p99_list, conc_recall_list)

    def run(self) -> float:
        """
        Returns:
            float: largest qps
        """
        return self._run_all_concurrencies_mem_efficient()

    def stop(self) -> None:
        pass

    def set_ef(self, ef) -> None:
        self.ef = ef
