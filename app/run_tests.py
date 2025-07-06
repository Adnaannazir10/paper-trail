#!/usr/bin/env python3
"""
Test runner script for the Paper Trail application.
"""

import subprocess
import sys
import os


def run_tests():
    """Run the test suite with coverage reporting."""
    
    # Change to the app directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Install test dependencies if not already installed
    print("Installing test dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements-test.txt"], check=True)
    
    # Run tests with coverage
    print("\nRunning tests with coverage...")
    test_command = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--cov=utils",
        "--cov=schemas",
        "--cov=api",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov",
        "--cov-report=xml:coverage.xml"
    ]
    
    result = subprocess.run(test_command)
    
    if result.returncode == 0:
        print("\n✅ All tests passed!")
        print("📊 Coverage report generated in htmlcov/")
        print("📄 XML coverage report generated as coverage.xml")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)


def run_unit_tests_only():
    """Run only unit tests (skip integration tests)."""
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    print("Running unit tests only...")
    test_command = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "-m", "not integration",
        "--cov=utils",
        "--cov-report=term-missing"
    ]
    
    result = subprocess.run(test_command)
    
    if result.returncode == 0:
        print("\n✅ All unit tests passed!")
    else:
        print("\n❌ Some unit tests failed!")
        sys.exit(1)


def run_integration_tests():
    """Run only integration tests."""
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    print("Running integration tests...")
    test_command = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "-m", "integration"
    ]
    
    result = subprocess.run(test_command)
    
    if result.returncode == 0:
        print("\n✅ All integration tests passed!")
    else:
        print("\n❌ Some integration tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "unit":
            run_unit_tests_only()
        elif sys.argv[1] == "integration":
            run_integration_tests()
        else:
            print("Usage: python run_tests.py [unit|integration]")
            print("  unit: Run only unit tests")
            print("  integration: Run only integration tests")
            print("  (no args): Run all tests with coverage")
    else:
        run_tests() 