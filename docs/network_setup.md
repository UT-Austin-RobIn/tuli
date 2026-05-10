Windows IP: `192.168.253.101`  
Linux IP: `192.168.253.201`  
Synology NAS: `192.168.253.1` 
Web interface for synology: `192.168.253.1:5000`.  
Synology: `robin`, `Robot123`.  


## Debugging Synology

We had seen an issue where synology wasn't being mounted with the command `sudo mount -t nfs 192.168.253.1:/volume1/tuli ~/synology-tuli`. Most likely, this error was because somehow the static IP of the linux machine had changed. It should be `192.168.253.201`.

1. Manually add ip addr through terminal on linux

```bash
sudo ip addr flush dev enp4s0
sudo ip addr add 192.168.253.201/24 dev enp4s0
sudo ip link set enp4s0 up
```

2. Then we logged into Synology assistant. We found two ways

- From terminal: 
`sudo arp-scan --interface=enp4s0 169.254.0.0/16`
It should return something like
```bash
Interface: enp4s0, type: EN10MB, MAC: 10:ff:e0:bb:34:1b, IPv4: 192.168.253.1
Starting arp-scan 1.9.7 with 65536 hosts ([https://github.com/royhills/arp-scan](https://github.com/royhills/arp-scan))
169.254.68.74	90:09:d0:6f:1f:09	(Unknown)
```
Them, we type the following on web browser (afair): `169.254.68.74`

- From app: download synology assistant


