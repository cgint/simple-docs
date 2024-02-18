import sys
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class TestRunner(FileSystemEventHandler):
    def on_any_event(self, event):
        if event.src_path.endswith(".py"):
            subprocess.run(["pytest", "tests/"])

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else '.'
    event_handler = TestRunner()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()
    try:
        while True:
            observer.join(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
