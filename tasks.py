#!/usr/bin/env python3
"""Development tasks - Alternative to Makefile"""

import subprocess
import sys
import argparse


def run(cmd):
    """Run a shell command"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        sys.exit(result.returncode)


def fix_formatting():
    """Fix code formatting"""
    print("🔧 Fixing formatting issues...")
    run("ruff check . --fix")
    run("ruff format .")
    print("✅ Formatting fixed!")


def check_formatting():
    """Check code formatting"""
    print("🔍 Checking formatting...")
    run("ruff check .")
    run("ruff format --check .")
    print("✅ Formatting check complete!")


def test():
    """Run tests"""
    print("🧪 Running tests...")
    run("pytest tests/test_utils.py -v")


def clean():
    """Clean cache files"""
    print("🧹 Cleaning cache files...")
    run("find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true")
    run("find . -type f -name '*.pyc' -delete")
    run("find . -type f -name '*.pyo' -delete")
    run("find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true")
    run("find . -type d -name '.pytest_cache' -exec rm -rf {} + 2>/dev/null || true")
    run("find . -type d -name '.ruff_cache' -exec rm -rf {} + 2>/dev/null || true")
    print("✅ Clean complete!")


def install_dev():
    """Install development dependencies"""
    print("📦 Installing development dependencies...")
    run("pip install ruff pytest pytest-cov")
    print("✅ Dependencies installed!")


def main():
    parser = argparse.ArgumentParser(description="Development tasks")
    parser.add_argument(
        "command",
        choices=["fix-formatting", "check-formatting", "test", "clean", "install-dev"],
        help="Command to run",
    )

    args = parser.parse_args()

    commands = {
        "fix-formatting": fix_formatting,
        "check-formatting": check_formatting,
        "test": test,
        "clean": clean,
        "install-dev": install_dev,
    }

    commands[args.command]()


if __name__ == "__main__":
    main()
