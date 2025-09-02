from collections import namedtuple

# Only use a subset of data and query for testing
DATA_SUBSET = None
QUERY_SUBSET = None

DatasetInfo = namedtuple(
    'DatasetInfo', [
        'name', 'data', 'query', 'dim', 'data_type', 'dist', 'filter_path', 'spec_label',
    ]
)

DATASETS = [
    DatasetInfo(
        name="cohere",
        data="/data/jsu068/bigann/bigann.128D.10M.euclidean.base.u8bin",
        query="/data/jsu068/bigann/bigann.128D.10K.euclidean.query.u8bin",
        dim=128,
        data_type="uint8",
        dist="L2",
        filter_path="/data/jsu068/bigann/bigann.10M.L50.zipf0.75.base.txt",
        spec_label=[3, 10, 15, 20, 37, 50],
    ),
    DatasetInfo(
        data="/data/jsu068/deep/deep.96D.10M.angular.base.fbin",
        query="/data/jsu068/deep/deep.96D.10K.angular.query.fbin",
        dim=96,
        data_type="float",
        dist="angular",
        filter_path="/data/jsu068/deep/bigann.10M.L50.zipf0.75.base.txt",
        spec_label=[3, 10, 15, 20, 37, 50],
    ),
    DatasetInfo(
        name="cohere",
        data="/data/zshen055/ANN/cohere/cohere_large_10m/base.fbin",
        query="/data/zshen055/ANN/cohere/cohere_large_10m/query.fbin",
        dim=768,
        data_type="float",
        dist="angular",
        filter_path="/data/jsu068/cohere/cohere.10M.L50.zipf0.75.txt",
        spec_label=[3, 10, 15, 20, 37, 50],
    ),
    DatasetInfo(
        name="openai",
        data="/data/jsu068/openai/openai.1536D.5M.angular.base.fbin",
        query="/data/jsu068/openai/openai.1536D.5M.angular.query.fbin",
        dim=1536,
        data_type="float",
        dist="angular",
        filter_path="/data/jsu068/openai/openai.5M.L50.zipf0.75.txt",
        spec_label=[3, 10, 15, 20, 37, 50],
    ),
    DatasetInfo(
        data="/data/jsu068/marco/embedding/marco.768D.10M.euclidean.fbin",
        query="/data/jsu068/marco/query/marco.768D.10K.euclidean.fbin",
        dim=768,
        data_type="float",
        dist="L2",
        filter_path="/data/jsu068/marco/embedding/marco.filter.base.10M.new.txt",
        spec_label=[i for i in range(48, 53)],
    ),
    DatasetInfo(
        data="/data/jsu068/yfcc/yfcc.192D.10M.euclidean.base.u8bin",
        query="/data/jsu068/yfcc/yfcc.192D.100K.euclidean.query.u8bin",
        dim=192,
        data_type="uint8",
        dist="L2",
        filter_path="/data/jsu068/yfcc/yfcc.filter.base.txt",
        spec_label=[29, 17, 28, 24, 106, 0],
    ),
]

hnsw_m = 32     # 96

# used for ef for HNSW query param
# BEAM_SIZES = [10, 15, 20, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210,
#               230, 250, 270, 290, 310, 330, 350, 370, 390, 410, 430, 450,
#               470, 490, 510, 530, 550, 570, 590, 610, 630, 650,]

# BEAM_SIZES = [10, 15, 20, 50, 110, 210, 410, 610]
BEAM_SIZES = [10, 50, 110, 410,]


TOP_K = 10
data_dir = "/data/jsu068"
DB = "annlib"
