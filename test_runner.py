#!/usr/bin/env python3
"""
Comprehensive Test Runner for Agent Personal Trainer Project

This script runs all tests in the project and generates detailed reports including:
- Test success/failure rates
- Individual test runtimes
- Detailed error information for failed tests
- Timestamped JSON reports
"""

import os
import sys
import json
import time
import traceback
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import pytest
from unittest.mock import patch
import asyncio
import argparse
import xml.etree.ElementTree as ET
import re

class TestRunner:
    """Comprehensive test runner for the entire project."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.test_logs_dir = self.project_root / "test_logs"
        self.test_logs_dir.mkdir(exist_ok=True)
        
        # Test results storage
        self.test_results: Dict[str, Dict[str, Any]] = {}
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.error_tests = 0
        self.coverage = 0
        
    def _clean_test_logs(self):
        """Remove all old log files from the test_logs directory."""
        print(f"🧹 Cleaning test logs in {self.test_logs_dir}...")
        cleaned_count = 0
        for file in self.test_logs_dir.glob('*.json'):
            try:
                file.unlink()
                cleaned_count += 1
            except OSError as e:
                print(f"Error deleting file {file}: {e}")
        print(f"  - Deleted {cleaned_count} log file(s).")
    
    def discover_tests(self) -> List[str]:
        """Discover all test files in the project."""
        test_files = []
        
        # Find all test files recursively
        for root, dirs, files in os.walk(self.project_root):
            # Skip virtual environments and cache directories
            if any(skip in root for skip in ['venv', '__pycache__', '.pytest_cache', 'node_modules']):
                continue
                
            for file in files:
                if file.startswith('test_') and file.endswith('.py'):
                    test_path = Path(root) / file
                    # Convert to relative path from project root
                    relative_path = test_path.relative_to(self.project_root)
                    test_files.append(str(relative_path))
        
        return sorted(test_files)
    
    def collect_test_names(self, test_file: str) -> List[str]:
        """Collect all test function names from a test file."""
        try:
            # Use pytest to collect test names without running them
            result = subprocess.run(
                [sys.executable, '-m', 'pytest', test_file, '--collect-only', '-q'],
                capture_output=True,
                text=True,
                cwd=self.project_root,
                timeout=30
            )
            
            test_names = []
            for line in result.stdout.split('\n'):
                if '::' in line and 'test_' in line:
                    # Extract test name from pytest output
                    parts = line.strip().split('::')
                    if len(parts) >= 3:
                        test_names.append(f"{parts[0]}::{parts[1]}::{parts[2]}")
            
            # If no tests found with the above method, try a different approach
            if not test_names:
                # Try running with more verbose output
                result = subprocess.run(
                    [sys.executable, '-m', 'pytest', test_file, '--collect-only', '-v'],
                    capture_output=True,
                    text=True,
                    cwd=self.project_root,
                    timeout=30
                )
                
                for line in result.stdout.split('\n'):
                    if '::' in line and 'test_' in line:
                        parts = line.strip().split('::')
                        if len(parts) >= 3:
                            test_names.append(f"{parts[0]}::{parts[1]}::{parts[2]}")
            
            return test_names
        except Exception as e:
            print(f"Error collecting tests from {test_file}: {e}")
            return []
    
    def run_single_test(self, test_name: str) -> Dict[str, Any]:
        """Run a single test and capture results."""
        start_time = time.time()
        result = {
            'test_name': test_name,
            'status': 'unknown',
            'runtime': 0.0,
            'error': None,
            'error_type': None,
            'error_traceback': None,
            'stdout': '',
            'stderr': ''
        }
        
        try:
            # Run the test with pytest
            cmd = [
                sys.executable, '-m', 'pytest', 
                test_name, 
                '-v', 
                '--tb=short',
                '--no-header',
                '--disable-warnings',
                '--cov=backend',
                '--cov-report=term'
            ]
            
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root,
                timeout=120  # 2 minute timeout per test
            )
            
            result['runtime'] = time.time() - start_time
            result['stdout'] = process.stdout
            result['stderr'] = process.stderr
            
            # Determine test status based on return code
            if process.returncode == 0:
                result['status'] = 'passed'
                self.passed_tests += 1
            else:
                result['status'] = 'failed'
                self.failed_tests += 1
                
                # Combine stdout and stderr to ensure we capture the traceback
                full_output = process.stdout + '\n' + process.stderr
                result['error_traceback'] = full_output
                
                # Extract a more specific error message
                error_summary = [line for line in full_output.split('\n') if line.strip().startswith('E ')]
                if error_summary:
                    result['error'] = error_summary[0].strip()
                else:
                    result['error'] = f"Test failed with return code {process.returncode}"
                result['error_type'] = 'TestFailure'
                
                # Extract coverage percentage from the output
                coverage_line = [line for line in full_output.split('\n') if 'TOTAL' in line]
                if coverage_line:
                    try:
                        coverage_str = coverage_line[0].split()[-1].replace('%', '')
                        self.coverage = int(coverage_str)
                    except (ValueError, IndexError):
                        pass # Ignore parsing errors
                
        except subprocess.TimeoutExpired as e:
            result['runtime'] = time.time() - start_time
            result['status'] = 'timeout'
            result['error'] = 'Test timed out after 120 seconds'
            result['error_type'] = 'TimeoutError'
            self.error_tests += 1
            
        except Exception as e:
            result['runtime'] = time.time() - start_time
            result['status'] = 'error'
            result['error'] = str(e)
            result['error_type'] = type(e).__name__
            result['error_traceback'] = traceback.format_exc()
            self.error_tests += 1
            
        self.total_tests += 1
        return result
    
    def run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests with proper markers."""
        print("🔍 Running integration tests...")
        
        # Use pytest to run all tests in the integration directory
        cmd = [
            sys.executable, '-m', 'pytest', 
            'backend/tests/integration/', 
            '-v', 
            '--tb=short',
            '--no-header',
            '--disable-warnings',
            '--cov=backend',
            '--cov-report=term'
        ]
        self._execute_test_run(cmd)
        return self.generate_report()

    def _execute_test_run(self, cmd: List[str]):
        """A helper function to execute a pytest command and parse results."""
        # Create a unique path for the JUnit XML report
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
        junit_xml_path = self.test_logs_dir / f"temp_junit_report_{timestamp}.xml"
        cmd.append(f"--junit-xml={junit_xml_path}")

        try:
            # Use Popen to allow for real-time streaming and capturing
            process = subprocess.Popen(
                cmd,
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1, # Line-buffered
                encoding='utf-8'
            )

            stdout_lines = []
            if process.stdout:
                for line in iter(process.stdout.readline, ''):
                    print(line, end='') # Stream to console
                    stdout_lines.append(line)
                process.stdout.close()
            
            process.wait(timeout=900) # Wait for the process to complete with a 15 min timeout
            
            full_output = "".join(stdout_lines)

            print("\n\n--- Captured Pytest Output for Debugging ---")
            print(full_output)
            print("--------------------------------------------\n")

            # --- Parse JUnit XML for detailed results ---
            if junit_xml_path.exists():
                self._parse_junit_xml(junit_xml_path)
                # Clean up the temp file
                os.remove(junit_xml_path)
            else:
                print("⚠️ JUnit XML report not found. Cannot determine individual test results.")
                # Clear test_results to indicate failure to parse details
                self.test_results.clear()

            # --- Parse Coverage from stdout ---
            coverage_line = [line for line in full_output.split('\n') if line.startswith('TOTAL')]
            if coverage_line:
                try:
                    # The line looks like: 'TOTAL 125 13 89%'
                    coverage_str = coverage_line[0].split()[-1].replace('%', '')
                    self.coverage = int(coverage_str)
                except (ValueError, IndexError):
                    self.coverage = -1 # Indicates parsing error
            else:
                # Fallback for when TOTAL line isn't present in the combined output
                final_line = full_output.strip().split('\n')[-1]
                if 'coverage' in final_line:
                    try:
                        # The line might look like: '... 14 passed ... 89% coverage'
                        coverage_match = re.search(r'(\d+)%\s+coverage', final_line)
                        if coverage_match:
                            coverage_str = coverage_match.group(1)
                            self.coverage = int(coverage_str)
                        else:
                            self.coverage = -2 # Indicates secondary parsing error
                    except(ValueError, IndexError):
                        self.coverage = -2 # Indicates secondary parsing error

        except subprocess.TimeoutExpired:
            print("Test run timed out.")
            if process:
                process.kill() # Ensure the process is terminated
            self.error_tests = self.total_tests - self.passed_tests
        except Exception as e:
            print(f"Error running test suite: {e}")
            self.error_tests = self.total_tests - self.passed_tests

    def _parse_junit_xml(self, xml_path: Path):
        """Parse the JUnit XML report to populate test results."""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            all_test_cases = list(root.iter('testcase'))
            total_tests_in_xml = len(all_test_cases)
            
            for i, testcase in enumerate(all_test_cases):
                class_name = testcase.attrib.get('classname', '').replace('.', '/')
                test_name_only = testcase.attrib.get('name', 'unknown_test')
                
                # Reconstruct the full test name in pytest format
                # e.g., backend/tests/unit/test_agent.py::TestClassName::test_method_name
                
                # Attempt to find the file path from classname
                file_path = ""
                parts = class_name.split('/')
                if len(parts) > 1:
                    # Assuming the path is something like backend/tests/unit/test_agent
                    potential_path = f"{'/'.join(parts)}.py"
                    if self.project_root.joinpath(potential_path).exists():
                        file_path = potential_path

                full_test_name = f"{file_path}::{testcase.attrib.get('classname', '')}::{test_name_only}"
                
                runtime = float(testcase.attrib.get('time', 0.0))
                status = 'passed'
                error_info = None
                error_traceback = None

                failure = testcase.find('failure')
                error = testcase.find('error')
                
                if failure is not None:
                    status = 'failed'
                    error_info = failure.attrib.get('message', 'No message')
                    error_traceback = failure.text
                elif error is not None:
                    status = 'error'
                    error_info = error.attrib.get('message', 'No message')
                    error_traceback = error.text
                
                self.test_results[full_test_name] = {
                    'test_name': full_test_name,
                    'status': status,
                    'runtime': runtime,
                    'error': error_info,
                    'error_traceback': error_traceback
                }
                
                # --- Live Progress Update ---
                progress = (i + 1) / total_tests_in_xml * 100
                status_icon = "✅" if status == 'passed' else "❌"
                print(f"\r[{progress:3.0f}%] {status_icon} Running: {test_name_only} ({runtime:.2f}s)", end="", flush=True)

            print() # Newline after progress bar is complete

        except ET.ParseError as e:
            print(f"Error parsing JUnit XML: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during XML parsing: {e}")

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests in the project, excluding long-running ones."""
        cmd = [
            sys.executable, '-m', 'pytest',
            'backend/tests/',
            '--ignore-glob', '*_long.py',
            '-v', '--tb=short', '--no-header',
            '--disable-warnings', '--cov=backend', '--cov-report=term'
        ]
        self._execute_test_run(cmd)
        return self.generate_report()

    def run_long_tests(self) -> Dict[str, Any]:
        """Run only the long-running tests."""
        print("🚀 Running LONG tests...")
        
        long_test_files = [str(p) for p in self.project_root.glob('backend/tests/integration/*_long.py')]
        
        if not long_test_files:
            print("No long tests found.")
            return self.generate_report()

        cmd = [
            sys.executable, '-m', 'pytest',
        ] + long_test_files + [
            '-v', '--tb=short', '--no-header',
            '--disable-warnings', '--cov=backend', '--cov-report=term'
        ]
        
        self._execute_test_run(cmd)
        return self.generate_report()

    def run_unit_tests(self) -> Dict[str, Any]:
        """Run only the unit tests."""
        print("🚀 Running UNIT tests...")
        cmd = [
            sys.executable, '-m', 'pytest',
            'backend/tests/unit/',
            '-v', '--tb=short', '--no-header',
            '--cov=backend', '--cov-report=term'
        ]
        self._execute_test_run(cmd)
        return self.generate_report()
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate a JSON report from the test results."""
        
        # Calculate summary stats
        self.passed_tests = sum(1 for r in self.test_results.values() if r['status'] == 'passed')
        self.failed_tests = sum(1 for r in self.test_results.values() if r['status'] == 'failed')
        self.error_tests = sum(1 for r in self.test_results.values() if r['status'] == 'error')
        self.total_tests = len(self.test_results)
        
        if self.total_tests > 0:
            success_rate = (self.passed_tests / self.total_tests) * 100
        else:
            success_rate = 0
            
        # Collect failed tests details
        failed_tests = []
        for test_name, result in self.test_results.items():
            if result['status'] != 'passed':
                failed_tests.append({
                    'test_name': test_name,
                    'status': result['status'],
                    'error': result.get('error'),
                    'error_type': result.get('error_type'),
                    'runtime': result['runtime'],
                    'traceback': result.get('error_traceback'),
                    'stdout': result.get('stdout', '')[:1000] + '...' if result.get('stdout') and len(result.get('stdout', '')) > 1000 else result.get('stdout'),
                    'stderr': result.get('stderr', '')[:1000] + '...' if result.get('stderr') and len(result.get('stderr', '')) > 1000 else result.get('stderr')
                })
        
        # Create runtime mapping
        test_runtimes = [{
            'test_name': test_name,
            'runtime': result['runtime']
        } for test_name, result in self.test_results.items()]
        
        # Sort tests by runtime descending
        sorted_runtimes = sorted(test_runtimes, key=lambda x: x['runtime'], reverse=True)
        
        total_runtime = sum(item['runtime'] for item in sorted_runtimes)
        average_runtime = total_runtime / self.total_tests if self.total_tests > 0 else 0

        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_tests': self.total_tests,
                'passed_tests': self.passed_tests,
                'failed_tests': self.failed_tests,
                'error_tests': self.error_tests,
                'success_rate_percentage': round(success_rate, 2),
                'coverage_percentage': self.coverage
            },
            'failed_tests': failed_tests,
            'test_runtimes': sorted_runtimes,
            'total_runtime': total_runtime,
            'average_runtime': average_runtime
        }
        return report
    
    def save_report(self, report: Dict[str, Any]) -> str:
        """Save the test report to a timestamped JSON file."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"test_report_{timestamp}.json"
        filepath = self.test_logs_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        return str(filepath)
    
    def print_summary(self, report: Dict[str, Any], report_path: str):
        """Print a human-readable summary of the test results."""
        summary = report['summary']
        
        print("\n" + "=" * 60)
        print(" TEST RUN SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {summary['total_tests']}")
        print(f"✅ Passed: {summary['passed_tests']}")
        print(f"❌ Failed: {summary['failed_tests']}")
        print(f"💥 Errors: {summary['error_tests']}")
        print(f"Success Rate: {summary['success_rate_percentage']}%")
        print(f"Coverage: {summary.get('coverage_percentage', 'N/A')}%")
        
        total_runtime = sum(t['runtime'] for t in report['test_runtimes'])
        average_runtime = total_runtime / summary['total_tests'] if summary['total_tests'] > 0 else 0
        
        print(f"Total Runtime: {total_runtime:.2f}s")
        print(f"Average Runtime: {average_runtime:.2f}s")
        
        if report['failed_tests']:
            print("\n❌ FAILED TESTS:")
            for test in report['failed_tests']:
                print(f"  • {test['test_name']}")
                print(f"    Status: {test['status']}")
                if test['error']:
                    print(f"    Error: {test['error']}")
                print(f"    Runtime: {test['runtime']:.2f}s")

        # Display slowest tests
        slowest_tests = report['test_runtimes'][:10]
        if slowest_tests:
            print("\n🐌 SLOWEST TESTS:")
            for test in slowest_tests:
                print(f"  • {test['test_name']}: {test['runtime']:.2f}s")
        
        print(f"\nReport saved to: {report_path}")

def main():
    """Main execution entry point."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Comprehensive Test Runner for Agent Personal Trainer",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--type',
        type=str,
        choices=['all', 'unit', 'integration', 'single', 'long'],
        default='all',
        help="Type of test to run: 'all', 'unit', 'integration', 'long', or 'single'."
    )
    parser.add_argument(
        '--name',
        type=str,
        help="Name of the test to run when --type is 'single'."
    )
    parser.add_argument(
        '--clean-logs',
        action='store_true',
        help="If set, clears all logs in the test_logs directory before the run."
    )
    args = parser.parse_args()

    print("🧪 Agent Personal Trainer - Comprehensive Test Runner")
    print("=" * 60)

    runner = TestRunner(project_root)

    # --- Clean logs if requested ---
    if args.clean_logs:
        runner._clean_test_logs()

    try:
        # Determine which tests to run
        if args.type == 'all':
            print("🎯 Running ALL tests (excluding long-running)...")
            report = runner.run_all_tests()
        elif args.type == 'unit':
            print("🎯 Running UNIT tests only...")
            report = runner.run_unit_tests()
        elif args.type == 'integration':
            print("🎯 Running INTEGRATION tests only...")
            report = runner.run_integration_tests()
        elif args.type == 'long':
            print("🎯 Running LONG tests only...")
            report = runner.run_long_tests()
        elif args.type == 'single':
            if not args.name:
                print("❌ Error: --name must be provided when --type is 'single'.")
            else:
                print(f"🎯 Running SINGLE test: {args.name}")
                report = runner.run_single_test(args.name)
        
        # Save report
        report_path = runner.save_report(report)
        
        # Print summary
        runner.print_summary(report, report_path)
        
        # Exit with appropriate code
        if report['summary']['failed_tests'] > 0 or report['summary']['error_tests'] > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n⚠️  Test run interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test runner failed: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 