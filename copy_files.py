#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File Copy Utility for Paper3
============================

This script copies essential files from paper1 and paper2 directories
into the paper3 structure for the integrated project.
"""

import os
import shutil
import sys
from tqdm import tqdm

def create_directory(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def copy_file(src, dst):
    """Copy a file and create destination directory if needed"""
    dst_dir = os.path.dirname(dst)
    create_directory(dst_dir)
    
    shutil.copy2(src, dst)
    return os.path.getsize(src)

def copy_directory(src, dst, ignore_patterns=None):
    """Copy a directory recursively"""
    if not os.path.exists(src):
        print(f"Source directory not found: {src}")
        return 0
    
    total_size = 0
    count = 0
    
    # Create destination if it doesn't exist
    create_directory(dst)
    
    # Walk through source directory
    for root, dirs, files in os.walk(src):
        # Create relative path to create the same structure in destination
        rel_path = os.path.relpath(root, src)
        dst_path = os.path.join(dst, rel_path)
        
        # Create necessary directories
        create_directory(dst_path)
        
        # Copy each file
        for file in files:
            # Skip files matching ignore patterns
            skip = False
            if ignore_patterns:
                for pattern in ignore_patterns:
                    if pattern in file:
                        skip = True
                        break
            
            if not skip:
                src_file = os.path.join(root, file)
                dst_file = os.path.join(dst_path, file)
                file_size = copy_file(src_file, dst_file)
                total_size += file_size
                count += 1
    
    return count, total_size

def main():
    # Base directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    # Source directories
    paper1_src = os.path.join(parent_dir, 'paper1')
    paper2_src = os.path.join(parent_dir, 'paper2')
    
    # Destination directories
    paper1_dst = os.path.join(script_dir, 'paper1')
    paper2_dst = os.path.join(script_dir, 'paper2')
    
    # Files/directories to ignore when copying
    ignore_patterns = [
        '__pycache__',
        '.git',
        '.vscode',
        '.DS_Store',
        'thumbs.db'
    ]
    
    print("Starting file copy process...")
    
    # Copy Paper 1 files
    print("\nCopying Paper 1 files...")
    count1, size1 = copy_directory(paper1_src, paper1_dst, ignore_patterns)
    print(f"✅ Copied {count1} files ({size1/1024/1024:.2f} MB) from Paper 1")
    
    # Copy Paper 2 files
    print("\nCopying Paper 2 files...")
    count2, size2 = copy_directory(paper2_src, paper2_dst, ignore_patterns)
    print(f"✅ Copied {count2} files ({size2/1024/1024:.2f} MB) from Paper 2")
    
    # Total statistics
    total_count = count1 + count2
    total_size = size1 + size2
    print(f"\nTotal: {total_count} files copied ({total_size/1024/1024:.2f} MB)")
    
    print("\nFile copy process completed successfully!")
    print(f"Now you can run integrated simulation using:\n  python {os.path.join(script_dir, 'main.py')}")

if __name__ == "__main__":
    main() 