import os
import time
import asyncio
from tqdm import tqdm
import numpy as np

from utils.config import *
from utils.mp_runner import MultiProcessingSearchRunner

import weaviate
from weaviate.classes.config import (
    Configure,
    Property,
    DataType,
    VectorDistances,
    VectorFilterStrategy
)
from weaviate.classes.query import Filter
from weaviate.config import AdditionalConfig, Timeout


class BatchImportError(Exception):
    """Raised when the batch import process encounters too many errors."""
    pass


##############################################################################
# Utility functions
##############################################################################

def load_points(file_path, dtype):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, 'rb') as f:
        n = int(np.fromfile(f, dtype=np.uint32, count=1)[0])
        dim = int(np.fromfile(f, dtype=np.uint32, count=1)[0])
        data = np.fromfile(f, dtype=dtype, count=n*dim)
        data = data.reshape(n, dim) if dtype == np.uint32 else data.reshape(
            n, dim).astype(np.float32)
    return data
# also use for load gt


def get_labels(file_path: str = None, n: int = None, label: str = None):
    if file_path is not None:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        labels = []
        with open(file_path, 'r') as f:
            for line in f:
                # ',' as separators of each label in points
                label_str = ',' + line.strip() + ','
                labels.append(label_str)
                # label_strs = line.strip().split(',')
                # label_ints = [int(s) for s in label_strs]
                # labels.append(label_ints)
        return labels
    return [label] * n

"""
def calculate_recall(gt_neighbors, results, top_k):
    recall = 0
    for i in range(len(gt_neighbors)):
        gt = np.sort(gt_neighbors[i])
        res = np.sort(results[i][:top_k])
        inter = len(np.intersect1d(gt, res, assume_unique=True))
        recall += inter / len(gt)
    return recall / len(gt_neighbors)
"""

def search(name, metric, query, query_label, k, ef):
    client = weaviate.connect_to_local(additional_config=AdditionalConfig(timeout=Timeout(query=60000)))
    collection = client.collections.get(name)
    try:
        async def async_query():
            return collection.query.hybrid(
                query='',
                vector=query,
                alpha=1,
                limit=k,
                filters=Filter.by_property("labels").like(f"*,{query_label},*"),
            )
        result = asyncio.run(async_query())
        hits = [res.properties.get("pid") for res in result.objects]
    except weaviate.exceptions.WeaviateQueryError as err:
        print(f"Query error for label {query_label}: {err}")
        hits = []
    finally:
        client.close()
    while len(hits) < k:
        hits.append(-1)
    return hits

##############################################################################
# Weaviate setup
##############################################################################

if __name__ == '__main__':

    THREADS = (os.cpu_count() or 1)
    print("THREADS", THREADS)
    print(
        "If you want to change the number of process used, OMP_NUM_THREADS and limits:cpus in docker-compose.yml should also both be changed."
    )

    os.makedirs("results", exist_ok=True)
    
    for info in DATASETS:
        data_name, base_path, query_path, dim, data_type, dist, filter_path, label_set = info
        dtype = np.uint8 if data_type == "uint8" else np.float32
        
        print("Loading data ...")
        data = load_points(base_path, dtype=dtype)
        if DATA_SUBSET is not None:
            data = data[:DATA_SUBSET]
        num_entities, dim = data.shape
        print("data.shape, ", data.shape)

        print("Loading labels ...")
        labels = get_labels(file_path=filter_path)
        if DATA_SUBSET is not None:
            labels = labels[:DATA_SUBSET]
        print("labels length: ", len(labels))

        print("Loading query ...")
        queries = load_points(query_path, dtype=dtype)
        if QUERY_SUBSET is not None:
            queries = queries[:QUERY_SUBSET]
        n_query, _ = queries.shape
        print("queries.shape, ", queries.shape)

        metric = VectorDistances.COSINE if dist == "angular" else VectorDistances.L2_SQUARED
    
##############################################################################
# Construction
##############################################################################
    
        print("Construction ...")
        for beam_size in BEAM_SIZES:
            # client = weaviate.Client("http://localhost:8080")
            client = weaviate.connect_to_local(additional_config=AdditionalConfig(timeout=Timeout(query=60000)))
            print(f"Connected to Weaviate. {client.is_ready()}")

            try:
                client.collections.delete_all()
                assert (not client.collections.list_all())
                print("All existing collections are dropped.\n")
            
                client.collections.create(
                    name=data_name,
                    vectorizer_config=Configure.Vectorizer.none(),
                    vector_index_config=Configure.VectorIndex.hnsw(
                        # quantizer=Configure.VectorIndex.Quantizer.bq(),
                        max_connections=hnsw_m,
                        ef=beam_size,
                        # ef=-1,
                        # dynamicEfMin=10,
                        # dynamicEfMax=210,
                        # dynamicEfFactor=20,
                        ef_construction=100,
                        distance_metric=metric,
                        filter_strategy=VectorFilterStrategy.ACORN,		# ver > 1.27.0
                        # filter_strategy=VectorFilterStrategy.SWEEPING,
                    ),
                    properties=[
                        Property(name="pid", data_type=DataType.INT),
                        # Property(name="vector", data_type=DataType.NUMBER_ARRAY),
                        Property(name="labels", index_filterable=False, data_type=DataType.TEXT),
                    ],
                )

                properties = [
                    {
                        "pid": i,
                        "labels": labels[i],
                    } for i in range(num_entities)
                ]

                start_time = time.time()

                collection = client.collections.get(name=data_name)

                # with collection.batch.fixed_size(batch_size=batch_size, concurrent_requests=batch_size) as batch:
                with collection.batch.dynamic() as batch:
                # with client.batch.dynamic() as batch:
                    for i in tqdm(range(num_entities)):
                        batch.add_object(
                            # collection=data_name,
                            # uuid=i,
                            properties=properties[i],
                            vector=data[i],
                        )
                        if batch.number_errors > 10:
                            raise BatchImportError("Exceeded maximum allowed errors during batch import")

                        # col = client.collections.get(name=data_name)
                        # failed_obj = col.batch.failed_objects
                        # if len(failed_obj) > 0:
                        #     print(failed_obj)
            except Exception as err:
                failed_objects = collection.batch.failed_objects
                if failed_objects:
                    print(f"Number of failed imports: {len(failed_objects)}")
                    print(f"First failed object: {failed_objects[0]}")
                print(err)
            finally:
                client.close()

            end_time = time.time()
            print(f"=== (ef={beam_size}) build index time = {(end_time - start_time):.4f}s", flush=True)

##############################################################################
# Filtered search
##############################################################################
            
            print("Querying ...")
            # collection = client.collections.get(data_name)

            for spec_label in label_set:
                query_gt = load_points(f"{data_dir}/{data_name}/N{num_entities}L{spec_label}.gt.bin", dtype=np.uint32)
                output_file = f"./results/weaviate.{data_name}.k{TOP_K}.zipf.L{spec_label}.csv"

                if not os.path.exists(output_file):
                    with open(output_file, "a") as f:
                        f.write("ef,qps,recall,p99_latency\n")

                query_labels = get_labels(n=queries.shape[0], label=spec_label)
                # search_result = [None] * n_query

                mpr = MultiProcessingSearchRunner(
                    func=search,
                    name=data_name,
                    metric=metric,
                    data=queries,
                    labels=query_labels,
                    gt=query_gt,
                    k=TOP_K,
                    concurrencies=[128, ],
                    duration=120,
                )

                result = mpr.run()
                print(f"=== query result: {result}")
                _, _, qps_list, p99_list, recall_list = result
                
                with open(output_file, "a") as f:
                    for qps, recall, p99_latency in zip(qps_list, recall_list, p99_list):
                        f.write(
                            f"{beam_size},{qps},{recall},{p99_latency}\n"
                        )

            """
            start_time = time.time()
            QUERY_BATCH = 10

            # parallel search
            with Manager() as manager:
                local_result = manager.list([None] * n_query)
                
                def parallel_search(i: int):
                    query_label = query_labels[i]
                    response = collection.query.hybrid(
                        query='',
                        vector=queries[i],
                        alpha=1,
                        limit=TOP_K,
                        filters=Filter.by_property("labels").like(f"*,{query_label},*"),
                    )
                    result = [res.properties.get("pid") for res in response.objects]
                    return (i, result)
            
                with ThreadPoolExecutor(max_workers=THREADS) as executor:
                    for i in tqdm(range(0, n_query, QUERY_BATCH), desc="Searching"):
                        futures = [executor.submit(parallel_search, j) for j in range(i, i + QUERY_BATCH)]
                        for future in as_completed(futures):
                            k, res = future.result()
                            local_result[k] = res
            
                # with ProcessPoolExecutor(max_workers=THREADS) as executor:
                #     list(tqdm(executor.map(parallel_search, range(len(queries))), total=len(queries), desc="Searching"))
                
                search_result = np.array(local_result)

                
            for i in tqdm(range(len(queries)), desc="Searching"):
                query_label = query_labels[i]

                # response = collection.query.fetch_objects(
                response = collection.query.hybrid(
                    query='',
                    vector=queries[i],
                    alpha=1,
                    limit=TOP_K,
                    filters=Filter.by_property("labels").like(f"*,{query_label},*"),
                )

                result = [res.properties.get("pid") for res in response.objects]
                search_result[i] = result

            end_time = time.time()
            total_time = end_time - start_time
            print(f"spec label: {spec_label} -" +
                " query latency = {:.4f}s".format(total_time), flush=True)

            qps = n_query / total_time
            query_gt = load_points(
                f"{data_dir}/{data_name}/N{num_entities}L{spec_label}.gt.bin", dtype=np.uint32)
            if QUERY_SUBSET is not None:
                query_gt = query_gt[:QUERY_SUBSET]
                
            recall = calculate_recall(query_gt, search_result, TOP_K)
            print("===average recall = {:.4f}".format(recall))
            print("===search latency = {:.4f}s".format(total_time))

            with open(output_file, "a") as f:
                f.write(f"{spec_label},{qps},{recall},{1/qps}\n")
            """
            
        client = weaviate.connect_to_local(additional_config=AdditionalConfig(timeout=Timeout(query=60000)))
        try:
            client.collections.delete_all()
            assert (not client.collections.list_all())
            print("All existing collections are dropped.\n")
        finally:
            client.close()
