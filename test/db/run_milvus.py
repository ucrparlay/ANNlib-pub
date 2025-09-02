import os
import time

import numpy as np
from tqdm import tqdm

from utils.config import *
from utils.mp_runner import MultiProcessingSearchRunner

from pymilvus import (
    connections,
    utility,
    DataType,
    FieldSchema,
    CollectionSchema,
    MilvusClient,
)

##############################################################################
# Utility functions
##############################################################################

# also use for load gt
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


# ',' as separators of each label in points
def get_labels(file_path: str = None, n: int = None, label: str = None):
    if file_path is not None:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        labels = []
        with open(file_path, 'r') as f:
            for line in f:
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
        gt = set(gt_neighbors[i])
        res = set(results[i][:top_k])
        recall += len(gt.intersection(res)) / len(gt)
    return recall / len(gt_neighbors)
"""

def get_index_params(index_type, metric):
    # m = 10 if dim == 100 else 8  # Because we need dim mod m == 0
    # assert dim % m == 0
    params = {
        "HNSW": {"M": 32, "efConstruction": 100},
    }
    if index_type not in params:
        raise Exception(f"Invalid index type {index_type}")

    index = {
        "index_type": index_type,
        "metric_type": metric,
        "params": params[index_type],
    }
    return index

"""
def get_query_params(index_type, metric):
    params = {
        "HNSW": [{
            "metric_type": metric,
            "params": {"ef": i}
        } for i in BEAM_SIZES],
        # "DISKANN": [{"search_list": i} for i in BEAM_SIZES],
    }
    if index_type not in params:
        raise Exception("Invalid index type")
    return params[index_type]
"""

def search(name, metric, query, query_label, k, ef):
    client = MilvusClient(uri="http://localhost:19530")
    result = client.search(
        collection_name=name,
        data=[query],
        limit=k,
        search_params={
            "HNSW": {
                "metric_type": metric,
                "params": {"ef": ef}
            },
            # "DISKANN": [{"search_list": i} for i in BEAM_SIZES],
        },
        output_fields=["labels"],
        filter=f'labels like "%,{query_label},%"',
    )
    # assert len(result) == 1
    hits = [hit.get('id') for hit in result[0]]
    while len(hits) < k:
        hits.append(-1)
    return hits

##############################################################################
# Milvus setup
##############################################################################

if __name__ == '__main__':

    THREADS = (os.cpu_count() or 1)
    print("THREADS", THREADS)
    print(
        "If you want to change the number of process used, OMP_NUM_THREADS and limits:cpus in docker-compose.yml should also both be changed."
    )

    os.makedirs("results", exist_ok=True)

    # connect to milvus
    client = MilvusClient(uri="http://localhost:19530")
    # connections.connect(alias=DB, host="localhost", port="19530")
    print("Connected to Milvus.")

    # create database (if not exists)
    # db.drop_database(db_name=DB, using=DB)
    # db.create_database(db_name=DB, using=DB)
    # db.list_database(using=DB)

    # remove all existing data
    connections.connect(host="localhost", port="19530")
    collections = utility.list_collections()
    for collection in collections:
        utility.drop_collection(collection)
    # collections = utility.list_collections(using=DB)
    # for collection in collections:
    #     utility.drop_collection(collection, using=DB)

    for info in DATASETS:

        # Modify to extract dimension from the dataset name if applicable
        # For example, if data_name is "bigann.128D.100M.euclidean", extract 128
        data_name, base_path, query_path, dim, data_type, dist, filter_path, label_set = info
        dtype = np.uint8 if data_type == "uint8" else np.float32

        print("Loading data ...")
        # data = load_points(os.path.join(data_dir, data_name), dtype=dtype)
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

        # metric = "IP" if "angular" in data_name else "L2"
        metric = "IP" if dist == "angular" else "L2"

##############################################################################
# Construction
##############################################################################

        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=False,
            ),
            FieldSchema(
                name="embeddings",
                dtype=DataType.FLOAT_VECTOR,
                dim=dim,
            ),
            FieldSchema(
                name="labels",
                # dtype=DataType.ARRAY,
                # element_type=DataType.INT32,
                # max_capacity=4096,
                dtype=DataType.VARCHAR,
                max_length=16384,
            ),
        ]
        schema = CollectionSchema(fields, "points with id and labels")
        """
        points = Collection(name=data_name, schema=schema,
                            using=DB, consistency_level="Strong")
        schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=True,
        )
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field(field_name="labels", datatype=DataType.ARRAY, element_type=DataType.INT32, max_capacity=4096)
        """

        client.create_collection(
            collection_name=data_name,
            # dimension=dim,
            schema=schema,
        )

        # base insertion
        print("Construction start")

        entities = [
            {
                "id": i,
                "embeddings": data[i],
                "labels": labels[i],
            } for i in range(num_entities)
        ]

        # max_message_size = 963347592
        # max_batch_size = max_message_size // (dim * 4)
        # batch_size = max_batch_size // 2  # Divide by 2 to be safe
        # batch_size = 10_000 if dim * num_entities > 1_280_000_000 else 100_000
        batch_size = 10_000

        for start in tqdm(range(0, num_entities, batch_size)):
            # entities_batch = [
            #     entities[0][start: start + batch_size],
            #     entities[1][start: start + batch_size],
            #     entities[2][start: start + batch_size],
            # ]
            # insert_result = points.insert(entities_batch)

            entities_batch = entities[start: start + batch_size]
            insert_result = client.insert(
                collection_name=data_name,
                data=entities_batch,
            )

            # print(f"insert_result: {start}\n",
            #       f"insert_count: {insert_result['insert_count']}, cost: {insert_result['cost']}")

        # points.flush()

        # use HNSW only
        # for index in INDEX_TYPES:
        index = "HNSW"
        print("Building index: ", index, flush=True)
        index_params = get_index_params(index_type=index, metric=metric)
        # search_params_list = get_query_params(index_type=index, metric=metric)
        print("index_params: ", index_params)
        formatted_index_params = "_".join(
            [f"{key}_{value}" for key, value in index_params["params"].items()]
        )

        start_time = time.time()

        # points.create_index("embeddings", index_params)
        # points.load()

        index_params = MilvusClient.prepare_index_params()

        index_params.add_index(
            field_name="embeddings",
            metric_type=metric,
            index_type=index,
            index_name="vector_index",
            params={
                "M": hnsw_m,
                "efConstruction": 100,
            }
        )

        index_params.add_index(
            field_name="labels",
            index_type='',
            index_name="label_index",
        )

        client.create_index(
            collection_name=data_name,
            index_params=index_params,
            sync=False,
        )

        client.load_collection(collection_name=data_name)
        indexing_prog = utility.index_building_progress(
            collection_name=data_name, index_name="vector_index", )
        loading_prog = utility.loading_progress(
            collection_name=data_name, )

        while indexing_prog.get("state") != "Finished" or loading_prog.get("loading_progress") != "100%":
            time.sleep(1)
            indexing_prog = utility.index_building_progress(
                collection_name=data_name, index_name="vector_index", )
            loading_prog = utility.loading_progress(
                collection_name=data_name, )

        end_time = time.time()

        print(indexing_prog)
        print(loading_prog)

        print("build index time = {:.4f}s".format(
            end_time - start_time), flush=True)

##############################################################################
# Filtered search
##############################################################################

        print("Querying ...")
        for spec_label in label_set:
            query_gt = load_points(f"{data_dir}/{data_name}/N{num_entities}L{spec_label}.gt.bin", dtype=np.uint32)
            output_file = f"./results/milvus.{data_name}.k{TOP_K}.zipf.L{spec_label}.csv"

            if not os.path.exists(output_file):
                with open(output_file, "a") as f:
                    f.write("ef,qps,recall,p99_latency\n")

            query_labels = get_labels(n=queries.shape[0], label=spec_label)
            # run_results = []

            mpr = MultiProcessingSearchRunner(
                func=search,
                name=data_name,
                metric=metric,
                data=queries,
                labels=query_labels,
                gt=query_gt,
                k=TOP_K,
                concurrencies=[128, ],
                duration=20,
            )

            for beam_size in BEAM_SIZES:
                mpr.set_ef(ef=beam_size)
                result = mpr.run()
                print(f"=== data {data_name} query with ef={beam_size}: {result}")
                _, _, qps_list, p99_list, recall_list = result
                
                with open(output_file, "a") as f:
                    for qps, recall, p99_latency in zip(qps_list, recall_list, p99_list):
                        f.write(
                            f"{beam_size},{qps},{recall},{p99_latency}\n"
                        )

            """
            for search_params in search_params_list:

                def search_point(ith_point):
                    # expr = f"labels contains {query_labels[ith_point]}"
                    query_label = query_labels[ith_point]

                    result = client.search(
                        collection_name=data_name,
                        data=[queries[ith_point]],
                        limit=TOP_K,
                        search_params=search_params,
                        output_fields=["labels"],
                        filter=f'labels like "%,{query_label},%"',
                    )

                    assert len(result) == 1
                    hits = result[0]
                    result_i = []
                    # distance = {}
                    for hit in hits:
                        result_i.append(hit.get('id'))
                        # if set(query_label) & set(hit.entity.get("labels")):
                        # result_i.append(hit.id)
                        # if len(result_i) == TOP_K:
                        #     break
                        # distance[hit.id] = hit.distance

                    while len(result_i) < TOP_K:
                        result_i.append(-1)

                    return ith_point, result_i

                # with ThreadPoolExecutor(max_workers=SIZE_OF_QUERY_POOL) as executor:

                print("===search_params: ", search_params)
                formatted_search_params = "_".join(
                    [f"{key}_{value}" for key, value in search_params.items()]
                )

                start_time = time.time()

                search_results = [None] * n_query
                for i in tqdm(range(len(queries)), desc="Searching"):
                    _, search_results[i] = search_point(i)

                # perform the search
                # manager = Manager()
                # search_results = manager.list([None] * n_query)

                # with Pool(processes=SIZE_OF_QUERY_POOL) as pool:
                #     for i, result in tqdm(
                #         pool.imap_unordered(
                #             search_point, [i for i in range(n_query)]),
                #         total=n_query,
                #         desc="Searching",
                #     ):
                #         search_results[i] = result

                end_time = time.time()

                total_time = end_time - start_time
                qps = len(queries) / total_time
                # Assuming ground truth is available for recall computation
                # You might need to adjust this part based on your actual ground truth data
                query_gt = load_points(
                    f"{data_dir}/{data_name}/N{num_entities}L{spec_label}.gt.bin", dtype=np.uint32)
                average_recall = calculate_recall(query_gt, search_results, TOP_K)
                print("===average recall = {:.4f}".format(average_recall))
                print("===search latency = {:.4f}s".format(total_time))

                run_results.append(
                    (
                        f"{index}_{formatted_index_params}_{formatted_search_params}",
                        qps,
                        average_recall,
                        total_time/len(queries),
                    )
                )

            with open(output_file, "a") as f:
                for config, qps, recall, avg_time in run_results:
                    f.write(
                        f"{config},{qps},{recall},{avg_time}\n"
                    )
            """
            # delete index
            # points.release()
            # points.drop_index()

        # remove all existing data
        # collections = utility.list_collections(using=DB)
        # for collection in collections:
        #     utility.drop_collection(collection, using=DB)
        utility.drop_collection(collection_name=data_name)

    # remove current database
    # db.drop_database(db_name=DB, using=DB)
