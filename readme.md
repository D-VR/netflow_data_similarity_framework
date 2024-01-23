# Requirements
This framework requires NetFlows in the binary NetFlow format nProbe https://www.ntop.org/products/netflow/nprobe/ 

e.g. NetFlow V1 Datasets: https://staff.itee.uq.edu.au/marius/NIDS_datasets/

# NetFlow Format

https://www.ntop.org/guides/nprobe/cli_options.html#netflow-v9-ipfix-format-t

NetFlow v9/IPFIX format [-T]

The following options can be used to specify the format:

[  7][Len 2] %L4_SRC_PORT                %sourceTransportPort        IPv4 source port

[  8][Len 4] %IPV4_SRC_ADDR              %sourceIPv4Address          IPv4 source address

[ 11][Len 2] %L4_DST_PORT                %destinationTransportPort   IPv4 destination port

[ 12][Len 4] %IPV4_DST_ADDR              %destinationIPv4Address     IPv4 destination address

[  1][Len 4] %IN_BYTES                   %octetDeltaCount            Incoming flow bytes (src->dst) [Aliased to %SRC_TO_DST_BYTES]

[  2][Len 4] %IN_PKTS                    %packetDeltaCount           Incoming flow packets (src->dst) [Aliased to %SRC_TO_DST_PKTS]

[ 23][Len 4] %OUT_BYTES                  %postOctetDeltaCount        Outgoing flow bytes (dst->src) [Aliased to %DST_TO_SRC_BYTES]

[ 24][Len 4] %OUT_PKTS                   %postPacketDeltaCount       Outgoing flow packets (dst->src) [Aliased to %DST_TO_SRC_PKTS]

[  4][Len 1] %PROTOCOL                   %protocolIdentifier         IP protocol byte

[  6][Len 1] %TCP_FLAGS                  %tcpControlBits             Cumulative of all flow TCP flags

[161][Len 4] %FLOW_DURATION_MILLISECONDS %flowDurationMilliseconds   Flow duration (msec)

Len x (Byte)

# Usage Examples

requires NetFlow in Queensland binary netflow format

## REAL DATA

### Similarity measures raw
`code` python3 syn_eval_cmd.py --ds1 "../../Datasets/Queensland_NetFlow/samples/real/NF-UNSW-NB15_0.csv" --ds2 "../../Datasets/Queensland_NetFlow/samples/real/NF-UNSW-NB15_1.csv" --expid 04-real --store_path "test_data/" 

### Aggregate results into csv file
`code` python3 syn_aggregate_cmd.py --load_path "test_data/04-real/similarity_raw" --ds_id 04-real --store_path "test_data/04-real_aggregate.csv"


## GENERATED DATA

### Syntax Check for filtering of correct NetFlow data
`code` python3 syntax_chk_cmd.py --load_path '../../Datasets/Queensland_NetFlow/synthetic/gpt2_data_clean/NF-UNSW-NB15_step-10000.csv' --ds_id 04-gpt2 --store_path "test_data/"

### Similarity measures raw
`code` python3 syn_eval_cmd.py --ds1 "../../Datasets/Queensland_NetFlow/samples/real/NF-UNSW-NB15_0.csv" --ds2 "test_data/04-gpt2/checked_data/NF-UNSW-NB15_step-10000.csv" --expid 04-gpt2 --store_path "test_data/" 

### Aggregate results into csv file
`code` python3 syn_aggregate_cmd.py --load_path "test_data/04-gpt2/similarity_raw" --ds_id 04-gpt --store_path "test_data/04-gpt2_aggregate.csv"
