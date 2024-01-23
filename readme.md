## Requirements
This framework requires NetFlows in the binary NetFlow format nProbe https://www.ntop.org/products/netflow/nprobe/ 
e.g. NetFlow V1 Datasets: https://staff.itee.uq.edu.au/marius/NIDS_datasets/

## NetFlow Format
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
