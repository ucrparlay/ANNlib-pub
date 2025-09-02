from pymilvus import (
	db,
    connections,
    utility,
)

# default
connections.connect(host="localhost", port="19530")
collections = utility.list_collections()
print(collections)
for collection in collections:
    utility.drop_collection(collection)

# annlib
DB = "annlib"
connections.connect(alias=DB, host="localhost", port="19530")

res = db.list_database(using=DB)
print(res)

if len(res) > 1:
	collections = utility.list_collections(using=DB)
	for collection in collections:
		utility.drop_collection(collection, using=DB)
	print("drop all collections")

	db.drop_database(db_name=DB, using=DB)
	print(f"drop database {DB}")
	db.list_database(using=DB)
