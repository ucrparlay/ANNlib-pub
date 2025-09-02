docker pull cr.weaviate.io/semitechnologies/weaviate:1.28.2 &&

docker run -d \
    -p 8080:8080 \
    -p 50051:50051 \
    --restart=unless-stopped \
    -e QUERY_DEFAULTS_LIMIT=10 \
    -e QUERY_MAXIMUM_RESULTS=650 \
    -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
    -e PERSISTENCE_DATA_PATH=/data/jsu068/weaviate \
    -e CLUSTER_HOSTNAME=node1 \
    -e GOMAXPROCS=224 \
    -e ASYNC_INDEXING=true \
    --name weaviate \
    cr.weaviate.io/semitechnologies/weaviate:1.28.2
