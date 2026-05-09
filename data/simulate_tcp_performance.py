import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def simulate_tcp_variant(variant="Reno", duration=60, packet_loss_rate=0.02, base_rtt=0.05):
    """
    Simulates the congestion window (cwnd) of a TCP connection over time.
    Returns times, cwnd, throughput, and jitter.
    """
    times = np.arange(0, duration, 0.1) # 100ms steps
    cwnd = []
    current_cwnd = 10
    ssthresh = 64
    
    # Trackers for simulation
    time_since_last_drop = 0
    
    for t in times:
        # Simulate a random packet drop event
        if np.random.random() < (packet_loss_rate * 0.1): 
            # Packet loss event
            if variant == "Reno":
                ssthresh = max(current_cwnd / 2, 2)
                current_cwnd = ssthresh # Fast recovery
            elif variant == "Cubic":
                ssthresh = max(current_cwnd * 0.7, 2) # Cubic reduces less (beta=0.7)
                current_cwnd = ssthresh
                time_since_last_drop = 0
        else:
            # Additive increase
            if variant == "Reno":
                if current_cwnd < ssthresh:
                    current_cwnd += 1 # Slow start (simplified)
                else:
                    current_cwnd += 1.0 / current_cwnd # AIMD
            elif variant == "Cubic":
                # Simplified cubic function: W(t) = C(t - K)^3 + W_max
                time_since_last_drop += 0.1
                # Cubic growth factor
                cubic_growth = 0.4 * (time_since_last_drop - 2)**3 + ssthresh
                current_cwnd = max(current_cwnd + 0.1, cubic_growth)
                
        # Impose a max network capacity (bottleneck)
        current_cwnd = min(current_cwnd, 120)
        cwnd.append(current_cwnd)
        
    # Calculate performance metrics
    throughput = [(w * 1500 * 8) / (base_rtt * 1_000_000) for w in cwnd] # Mbps
    rtt_variations = np.random.normal(base_rtt, base_rtt * 0.2, len(times)) # Jitter simulation
    
    return times, np.array(cwnd), np.array(throughput), rtt_variations

def run_simulation():
    if not os.path.exists("reports"):
        os.makedirs("reports")

    print("[*] Running TCP Reno vs TCP Cubic Simulation...")
    
    # Run simulations under identical network conditions
    t, reno_cwnd, reno_tp, reno_rtt = simulate_tcp_variant("Reno", packet_loss_rate=0.05)
    t, cubic_cwnd, cubic_tp, cubic_rtt = simulate_tcp_variant("Cubic", packet_loss_rate=0.05)
    
    # 1. Generate Visual Graphs
    plt.figure(figsize=(12, 8))
    
    # Congestion Window Plot
    plt.subplot(2, 1, 1)
    plt.plot(t, reno_cwnd, label="TCP Reno (AIMD)", color='blue', alpha=0.8)
    plt.plot(t, cubic_cwnd, label="TCP Cubic", color='red', alpha=0.8)
    plt.title("TCP Congestion Window Size Over Time (Simulated 5% Packet Loss)")
    plt.ylabel("Window Size (Packets)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Throughput Plot
    plt.subplot(2, 1, 2)
    plt.plot(t, reno_tp, label="TCP Reno Throughput", color='blue', alpha=0.6)
    plt.plot(t, cubic_tp, label="TCP Cubic Throughput", color='red', alpha=0.6)
    plt.title("Estimated Throughput (Mbps)")
    plt.xlabel("Time (Seconds)")
    plt.ylabel("Throughput (Mbps)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig("reports/tcp_congestion_comparison.png")
    print("[+] Graph saved to reports/tcp_congestion_comparison.png")
    
    # 2. Generate Quantitative Report
    report_data = {
        "Metric": ["Average Throughput (Mbps)", "Max Throughput (Mbps)", "Average RTT (ms)", "Jitter (ms)", "Avg Congestion Window"],
        "TCP Reno": [
            f"{np.mean(reno_tp):.2f}",
            f"{np.max(reno_tp):.2f}",
            f"{np.mean(reno_rtt)*1000:.2f}",
            f"{np.std(reno_rtt)*1000:.2f}",
            f"{np.mean(reno_cwnd):.1f}"
        ],
        "TCP Cubic": [
            f"{np.mean(cubic_tp):.2f}",
            f"{np.max(cubic_tp):.2f}",
            f"{np.mean(cubic_rtt)*1000:.2f}",
            f"{np.std(cubic_rtt)*1000:.2f}",
            f"{np.mean(cubic_cwnd):.1f}"
        ]
    }
    
    df = pd.DataFrame(report_data)
    df.to_csv("reports/tcp_performance_metrics.csv", index=False)
    print("\n[+] Performance Metrics Summary:")
    print("-" * 60)
    print(df.to_string(index=False))
    print("-" * 60)
    print("[+] Tabular data saved to reports/tcp_performance_metrics.csv")

if __name__ == "__main__":
    run_simulation()
