from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess
import time
import os
import platform

class ChangeHandler(FileSystemEventHandler):
    def __init__(self):
        self.ml_process = None
        self.dashboard_process = None
        self.price_updater_process = None
        self.fetch_coinex_process = None
        self.scheduler_combined_process = None

        self.start_dashboard()
        self.start_price_updater()
        self.start_fetch_coinex()
        self.start_scheduler_combined()
        self.restart_main_ml()

    def start_dashboard(self):
        print("🚀 اجرای dashboard.py ...")
        if platform.system() == "Windows":
            self.dashboard_process = subprocess.Popen(["start", "cmd", "/k", "python dashboard.py"], shell=True)
        else:
            self.dashboard_process = subprocess.Popen(["gnome-terminal", "--", "python3", "dashboard.py"])

    def start_price_updater(self):
        print("🔁 اجرای price_updater.py ...")
        if platform.system() == "Windows":
            self.price_updater_process = subprocess.Popen(["start", "cmd", "/k", "python price_updater.py"], shell=True)
        else:
            self.price_updater_process = subprocess.Popen(["gnome-terminal", "--", "python3", "price_updater.py"])

    def start_fetch_coinex(self):
        print("📡 اجرای fetch_coinex.py ...")
        if platform.system() == "Windows":
            self.fetch_coinex_process = subprocess.Popen(["start", "cmd", "/k", "python data/fetch_coinex.py"], shell=True)
        else:
            self.fetch_coinex_process = subprocess.Popen(["gnome-terminal", "--", "python3", "data/fetch_coinex.py"])

    def start_scheduler_combined(self):
        print("📰 اجرای scheduler_combined.py ...")
        if platform.system() == "Windows":
            self.scheduler_combined_process = subprocess.Popen(["start", "cmd", "/k", "python scheduler_combined.py"], shell=True)
        else:
            self.scheduler_combined_process = subprocess.Popen(["gnome-terminal", "--", "python3", "scheduler_combined.py"])

    def restart_main_ml(self):
        if self.ml_process:
            self.ml_process.terminate()
        print("♻️ ری‌استارت main_ml.py ...")
        self.ml_process = subprocess.Popen(["python", "main_ml.py"])

    def on_modified(self, event):
        # اگر فایل simulation_trades.json تغییر کرد، ری‌استارت نکن
        if "simulation_trades.json" in event.src_path:
            return

        if any(event.src_path.endswith(ext) for ext in [".py", ".json", ".pkl"]):
            print(f"📄 تغییر شناسایی شد: {event.src_path}")
            self.restart_main_ml()

if __name__ == "__main__":
    path = "."  # مسیر پروژه
    event_handler = ChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, path=path, recursive=True)
    observer.start()
    print("👁 مانیتورینگ تغییرات فایل‌ها شروع شد...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("🛑 مانیتورینگ متوقف شد.")
    observer.join()
