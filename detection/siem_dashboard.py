import time
import csv
import random
import os
from datetime import datetime
import threading

class SIEMDashboard:
    def __init__(self, ai_log_path="reports/alert_log.csv"):
        self.ai_log_path = ai_log_path
        self.syslog_events = []
        self.correlated_alerts = []
        self.running = True

        # Ensure the reports directory exists
        if not os.path.exists("reports"):
            os.makedirs("reports")
            
        # Create a dummy alert log if it doesn't exist to prevent crashes
        if not os.path.exists(self.ai_log_path):
            with open(self.ai_log_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Source IP", "Destination IP", "Prediction", "Confidence"])

    def generate_simulated_syslog(self):
        """Simulates incoming Syslog messages from Cisco Packet Tracer devices"""
        devices = ["ASA-FW01", "DS1-CORE", "AS1-ACCESS", "AS2-ACCESS"]
        messages = [
            ("%SEC-6-IPACCESSLOGP: list OUTSIDE-IN denied tcp", "Warning"),
            ("%SEC-4-IPACCESSLOGDP: drop from 192.168.10.45", "Warning"),
            ("%SSH-5-SSH2_SESSION: SSH2 Session login failed", "High"),
            ("%LINK-3-UPDOWN: Interface GigabitEthernet0/1, changed state to down", "Critical"),
            ("%LINK-3-UPDOWN: Interface GigabitEthernet0/1, changed state to up", "Info"),
            ("%SNMP-3-AUTHFAIL: Authentication failure for SNMP req", "High")
        ]
        
        while self.running:
            time.sleep(random.uniform(2.0, 5.0))
            device = random.choice(devices)
            msg, severity = random.choice(messages)
            src_ip = f"192.168.10.{random.randint(10, 50)}"
            
            syslog_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "device": device,
                "message": f"{msg} (src: {src_ip})",
                "severity": severity,
                "ip": src_ip
            }
            self.syslog_events.append(syslog_entry)
            
            # Keep log size manageable
            if len(self.syslog_events) > 50:
                self.syslog_events.pop(0)

    def read_ai_alerts(self):
        """Reads the AI predictions from Phase 3"""
        alerts = []
        try:
            with open(self.ai_log_path, mode='r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    alerts.append(row)
        except Exception:
            pass
        return alerts[-20:] if len(alerts) > 20 else alerts

    def correlate_threats(self, ai_alerts):
        """Correlates Syslog events with AI ML predictions"""
        self.correlated_alerts = []
        
        # Look for matching IPs within the last few minutes
        for ai in ai_alerts:
            # Skip normal traffic unless we see suspicious syslog
            if ai.get('Prediction') == 'Normal':
                continue
                
            ai_ip = ai.get('Source IP', '')
            matched_syslogs = [s for s in self.syslog_events if s['ip'] == ai_ip]
            
            if matched_syslogs:
                # High severity - AI detected an anomaly AND Firewall/Switch generated a syslog error
                self.correlated_alerts.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "threat": f"CRITICAL: {ai.get('Prediction')} confirmed by network logs",
                    "ip": ai_ip,
                    "evidence": f"AI Confidence {ai.get('Confidence', 'N/A')} + {len(matched_syslogs)} syslog alerts"
                })
            else:
                # Medium severity - AI detected something, but no syslog confirmation yet
                self.correlated_alerts.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "threat": f"WARNING: AI detected {ai.get('Prediction')} (Unverified)",
                    "ip": ai_ip,
                    "evidence": f"AI Confidence {ai.get('Confidence', 'N/A')}"
                })

    def display_dashboard(self):
        """Renders the terminal UI"""
        os.system('cls' if os.name == 'nt' else 'clear')
        print("="*80)
        print("    AI-ENHANCED SIEM DASHBOARD (PHASE 4 CORRELATION ENGINE)    ")
        print("="*80)
        
        print("\n[+] RECENT SYSLOG EVENTS (from Packet Tracer Simulation):")
        print("-" * 80)
        for log in self.syslog_events[-5:]:
            print(f"[{log['timestamp']}] {log['device']} | {log['severity']} | {log['message']}")
            
        print("\n[+] RECENT AI ML ANOMALY DETECTIONS (from Phase 3):")
        print("-" * 80)
        ai_alerts = self.read_ai_alerts()
        for ai in ai_alerts[-5:]:
            print(f"[{ai.get('Timestamp', 'N/A')}] Source: {ai.get('Source IP', 'N/A')} -> {ai.get('Prediction', 'N/A')} (Conf: {ai.get('Confidence', 'N/A')})")
            
        print("\n" + "="*80)
        print("    CORRELATED THREAT INTELLIGENCE    ")
        print("="*80)
        self.correlate_threats(ai_alerts)
        if not self.correlated_alerts:
            print("No correlated threats detected. Network is currently secure.")
        else:
            for alert in self.correlated_alerts[-5:]:
                color = "\033[91m" if "CRITICAL" in alert['threat'] else "\033[93m"
                reset = "\033[0m"
                print(f"{color}[{alert['timestamp']}] IP: {alert['ip']} | {alert['threat']}{reset}")
                print(f"    Evidence: {alert['evidence']}")
        
        print("\nPress Ctrl+C to exit...")

    def run(self):
        syslog_thread = threading.Thread(target=self.generate_simulated_syslog)
        syslog_thread.daemon = True
        syslog_thread.start()
        
        try:
            while True:
                self.display_dashboard()
                time.sleep(3)
        except KeyboardInterrupt:
            self.running = False
            print("\nShutting down SIEM Dashboard...")

if __name__ == "__main__":
    siem = SIEMDashboard()
    siem.run()
