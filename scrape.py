#!/usr/bin/env python3
# broadcast_dump.py

from scapy.all import sniff, IP, UDP, Raw, conf

# Broadcast address to watch
BCAST_IP = "192.168.178.255"

def dump_broadcast(pkt):
    # Only look at IPv4 UDP packets destined for our broadcast IP
    if IP in pkt and UDP in pkt and pkt[IP].dst == BCAST_IP:
        src = pkt[IP].src
        sport = pkt[UDP].sport
        dport = pkt[UDP].dport
        print(f"\n← Broadcast from {src}:{sport} to {BCAST_IP}:{dport}")
        if Raw in pkt:
            # dump payload as both repr and str
            data = pkt[Raw].load
            print(f"   Payload ({len(data)} bytes):\n{data!r}")
        else:
            print("   (no payload)")

if __name__ == "__main__":
    # suppress Scapy IPv6 warning
    conf.ipv6_enabled = False

    print(f"Listening for UDP broadcasts to {BCAST_IP}… (CTRL-C to stop)")
    # BPF filter: only IPv4 UDP packets destined for BCAST_IP
    bpf = f"udp and dst host {BCAST_IP}"
    sniff(filter=bpf, prn=dump_broadcast, store=False)

