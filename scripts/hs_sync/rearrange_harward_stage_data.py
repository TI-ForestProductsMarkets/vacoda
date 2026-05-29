cd /storage/vacoda/data/external/harvard_comtrade/as_reported

mkdir -p raw_parquet

for hsdir in raw_parquet_partitioned/hs_partition=*; do
    hs="${hsdir##*=}"

    for yeardir in "$hsdir"/year_partition=*; do
        year="${yeardir##*=}"

        mkdir -p "raw_parquet/$hs/$year"

        # move or copy parquet files
        cp "$yeardir"/*.parquet "raw_parquet/$hs/$year/"
    done
done