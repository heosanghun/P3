#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simulation Monitoring Script
============================

This script executes main.py and monitors its progress in real-time.
"""

import os
import sys
import time
import subprocess
import threading
import psutil
import datetime
from tqdm import tqdm
import colorama
from colorama import Fore, Back, Style

# Initialize colorama
colorama.init()

# Constants
GREEN = Fore.GREEN
YELLOW = Fore.YELLOW
CYAN = Fore.CYAN
RED = Fore.RED
BLUE = Fore.BLUE
MAGENTA = Fore.MAGENTA
RESET = Style.RESET_ALL

class SimulationMonitor:
    def __init__(self, script_path="main.py"):
        self.script_path = script_path
        self.start_time = None
        self.process = None
        self.is_running = False
        self.log_file = f"simulation_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self.results_dir = None
        self.has_error = False
        
    def log(self, message, color=CYAN):
        """Log a message to console and file"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        console_msg = f"{color}[{timestamp}] {message}{RESET}"
        file_msg = f"[{timestamp}] {message}"
        
        print(console_msg)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(file_msg + "\n")
    
    def monitor_cpu_memory(self):
        """Monitor CPU and memory usage of the process"""
        if not self.process:
            return
        
        try:
            p = psutil.Process(self.process.pid)
            
            # Create progress bar for CPU and memory
            cpu_bar = tqdm(total=100, desc="CPU Usage", bar_format="{desc}: {percentage:3.1f}%|{bar}| {n:.1f}/{total:.1f}%", leave=False)
            mem_bar = tqdm(total=p.memory_info().rss // 1024 // 1024 + 100, desc="Memory", bar_format="{desc}: {n:4d}MB", leave=False)
            
            # Update CPU and memory usage every second
            while self.is_running:
                try:
                    # Get CPU and memory usage
                    cpu_percent = p.cpu_percent(interval=1)
                    memory_mb = p.memory_info().rss // 1024 // 1024
                    
                    # Update progress bars
                    cpu_bar.n = cpu_percent
                    mem_bar.n = memory_mb
                    
                    cpu_bar.refresh()
                    mem_bar.refresh()
                    
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
                except Exception as e:
                    self.log(f"Error monitoring resources: {str(e)}", RED)
                    break
                
                time.sleep(1)
            
            # Close progress bars
            cpu_bar.close()
            mem_bar.close()
            
        except Exception as e:
            self.log(f"Failed to monitor resources: {str(e)}", RED)
    
    def scan_for_results_dir(self):
        """Scan for newly created results directory"""
        before = set(os.listdir("results")) if os.path.exists("results") else set()
        
        # Wait a bit to let the simulation create the directory
        time.sleep(3)
        
        # Check every 5 seconds for new directories
        while self.is_running:
            if os.path.exists("results"):
                after = set(os.listdir("results"))
                new_dirs = after - before
                
                for dir_name in new_dirs:
                    if dir_name.startswith("integrated_run_"):
                        full_path = os.path.join("results", dir_name)
                        if os.path.isdir(full_path):
                            self.results_dir = full_path
                            self.log(f"Detected results directory: {self.results_dir}", GREEN)
                            return
            
            time.sleep(5)
    
    def count_files_in_dir(self, directory):
        """Count files in a directory recursively"""
        count = 0
        if os.path.exists(directory):
            for root, dirs, files in os.walk(directory):
                count += len(files)
        return count
    
    def monitor_results(self):
        """Monitor the results directory"""
        if not self.results_dir:
            return
        
        # Wait for directories to be created
        time.sleep(5)
        
        # Check for generated files every 5 seconds
        with tqdm(desc="Generated Files", bar_format="{desc}: {n:4d} files", leave=False) as pbar:
            last_count = 0
            
            while self.is_running:
                current_count = self.count_files_in_dir(self.results_dir)
                
                # Update progress if count changed
                if current_count > last_count:
                    pbar.update(current_count - last_count)
                    last_count = current_count
                    
                    # Check for visualization files
                    viz_dir = os.path.join(self.results_dir, "comparative_analysis", "visualizations")
                    if os.path.exists(viz_dir) and os.listdir(viz_dir):
                        self.log(f"Visualizations are being generated in: {viz_dir}", GREEN)
                
                time.sleep(5)
    
    def run(self):
        """Run the simulation and monitor its progress"""
        self.log("Starting simulation monitor...", BLUE)
        self.log(f"Executing: python {self.script_path}", YELLOW)
        
        # Record start time
        self.start_time = time.time()
        
        try:
            # Start the process
            self.process = subprocess.Popen(
                [sys.executable, self.script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1
            )
            self.is_running = True
            
            # Start monitoring threads
            resource_thread = threading.Thread(target=self.monitor_cpu_memory)
            resource_thread.daemon = True
            resource_thread.start()
            
            results_dir_thread = threading.Thread(target=self.scan_for_results_dir)
            results_dir_thread.daemon = True
            results_dir_thread.start()
            
            # Show elapsed time
            elapsed_bar = tqdm(desc="Elapsed Time", bar_format="{desc}: {elapsed}", leave=False)
            
            # Process output in real-time
            for line in self.process.stdout:
                line = line.strip()
                if line:
                    print(line)
                    with open(self.log_file, "a", encoding="utf-8") as f:
                        f.write(line + "\n")
                
                # Update elapsed time
                elapsed_bar.update(0)
                
                # Check if results directory is found, start monitoring
                if self.results_dir and not hasattr(self, 'results_thread'):
                    self.results_thread = threading.Thread(target=self.monitor_results)
                    self.results_thread.daemon = True
                    self.results_thread.start()
            
            # Process any errors
            for line in self.process.stderr:
                line = line.strip()
                if line:
                    self.has_error = True
                    print(f"{RED}{line}{RESET}")
                    with open(self.log_file, "a", encoding="utf-8") as f:
                        f.write(f"ERROR: {line}\n")
            
            # Wait for process to complete
            self.process.wait()
            self.is_running = False
            
            # Close elapsed time bar
            elapsed_bar.close()
            
            # Calculate total runtime
            end_time = time.time()
            runtime = end_time - self.start_time
            
            # Print summary
            self.log("="*50, BLUE)
            self.log("Simulation completed!", GREEN if not self.has_error else RED)
            self.log(f"Total Runtime: {datetime.timedelta(seconds=int(runtime))}", YELLOW)
            
            if self.results_dir:
                self.log(f"Results saved to: {self.results_dir}", GREEN)
                
                # Count files in results directory
                file_count = self.count_files_in_dir(self.results_dir)
                self.log(f"Generated {file_count} files", CYAN)
                
                # Check for visualizations
                viz_dir = os.path.join(self.results_dir, "comparative_analysis", "visualizations")
                if os.path.exists(viz_dir):
                    viz_count = len(os.listdir(viz_dir))
                    self.log(f"Generated {viz_count} visualization charts", MAGENTA)
                    
                    # List visualization files
                    self.log("Generated charts:", CYAN)
                    for viz_file in os.listdir(viz_dir):
                        self.log(f"  - {viz_file}", CYAN)
            
            return self.process.returncode
            
        except KeyboardInterrupt:
            self.log("Simulation interrupted by user.", YELLOW)
            if self.process:
                self.process.terminate()
            return 1
        except Exception as e:
            self.log(f"Error running simulation: {str(e)}", RED)
            return 1

if __name__ == "__main__":
    # Create a fancy banner
    print("\n" + "="*80)
    print(f"{CYAN}SIMULATION MONITOR - PAPER 3{RESET}".center(80))
    print(f"{YELLOW}Real-time progress tracking and performance monitoring{RESET}".center(80))
    print("="*80 + "\n")
    
    # Run the monitor
    monitor = SimulationMonitor("main.py")
    sys.exit(monitor.run()) 