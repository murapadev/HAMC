#!/usr/bin/env python3
import pytest
import sys
import os
from datetime import datetime

def main():
    """Run test suite with coverage reporting."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.join("test_reports", timestamp)
    os.makedirs(report_dir, exist_ok=True)
    
    # Run tests with coverage
    args = [
        "--verbose",
        "--cov=hamc",
        "--cov-report=html:" + os.path.join(report_dir, "coverage"),
        "--cov-report=term-missing",
        "--junitxml=" + os.path.join(report_dir, "test_results.xml"),
        "tests"
    ]
    
    return pytest.main(args)

if __name__ == "__main__":
    sys.exit(main())