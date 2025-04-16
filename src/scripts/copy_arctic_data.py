from arcticdb import Arctic

library = "prices"
symbol_regex = "^ig-trading"

# source_uri = "lmdb:///home/rory/dev/airflow/test/arctic.db"
target_uri = (
    "s3://s3.us-east-1.amazonaws.com:tradingo-store?aws_auth=true&path_prefix=prod"
)
source_uri = "lmdb:///home/rory/dev/tradingo-plat/data/prod/tradingo.db"


def sync_symbols(
    library,
    symbol_regex,
    source_uri,
    target_uri,
):
    source_a = Arctic(source_uri)
    target_a = Arctic(target_uri)

    source_lib = source_a.get_library(library)
    target_lib = target_a.get_library(library, create_if_missing=True)

    for sym in source_lib.list_symbols(regex=symbol_regex):
        data = source_lib.read(sym)
        target_lib.write(sym, data.data)
