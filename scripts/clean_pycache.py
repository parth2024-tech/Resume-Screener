import os
import shutil

def cleanup():
    """Locate and delete all __pycache__ folders and compiled python files."""
    deleted_dirs = 0
    deleted_files = 0
    
    for root, dirs, files in os.walk("."):
        # Ignore virtual environment directory
        if "venv" in root or ".git" in root:
            continue
            
        for d in list(dirs):
            if d == "__pycache__":
                dir_path = os.path.join(root, d)
                shutil.rmtree(dir_path)
                dirs.remove(d)
                deleted_dirs += 1
                
        for f in files:
            if f.endswith(".pyc") or f.endswith(".pyo"):
                file_path = os.path.join(root, f)
                os.remove(file_path)
                deleted_files += 1
                
    print(f"🧹 Cleaned {deleted_dirs} __pycache__ directories and {deleted_files} compiled files.")

if __name__ == "__main__":
    cleanup()
