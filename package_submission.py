import tarfile
import os

def main():
    files_to_pack = [
        ("submission_v16.tar.gz", "V16 Submission Package"),
        ("submission.tar.gz", "General Submission Package")
    ]
    
    buddy_dir = "buddy_model"
    
    for filename, desc in files_to_pack:
        print(f"Creating {desc}: {filename}...")
        with tarfile.open(filename, "w:gz") as tar:
            # Add main.py as main.py at root of the archive
            main_path = os.path.join(buddy_dir, "main.py")
            tar.add(main_path, arcname="main.py")
            
            # Add orbit_lite directory recursively, excluding __pycache__
            orbit_lite_dir = os.path.join(buddy_dir, "orbit_lite")
            for root, dirs, files in os.walk(orbit_lite_dir):
                # Modify dirs in place to exclude __pycache__ from os.walk
                if "__pycache__" in dirs:
                    dirs.remove("__pycache__")
                
                for file in files:
                    full_path = os.path.join(root, file)
                    # Get path relative to buddy_model
                    arcname = os.path.relpath(full_path, buddy_dir)
                    tar.add(full_path, arcname=arcname)
                    
        print(f"Created {filename} successfully.")

if __name__ == "__main__":
    main()
