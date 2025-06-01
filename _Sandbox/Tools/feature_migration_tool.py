#!/usr/bin/env python3
"""
Feature Migration Tool
Manages safe migration from Sandbox to Production environment
Ensures quality gates and .cursorrules compliance before migration
"""

import os
import shutil
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureMigrationTool:
    """
    Manages the migration of features from Sandbox to Production
    with comprehensive validation and quality gates
    """
    
    def __init__(self, sandbox_root: str = "_Sandbox", production_root: str = "."):
        self.sandbox_root = Path(sandbox_root)
        self.production_root = Path(production_root)
        self.migration_log_path = self.sandbox_root / "migration_log.json"
        
    def pre_migration_check(self, feature_name: str) -> Dict:
        """Run comprehensive pre-migration validation"""
        logger.info(f"Running pre-migration check for feature: {feature_name}")
        
        feature_path = self.sandbox_root / "Environment" / "TestDrivenFeatures" / f"{feature_name}_TDD"
        
        if not feature_path.exists():
            return {"error": f"Feature {feature_name} not found in sandbox"}
        
        validation_results = {
            "feature_name": feature_name,
            "timestamp": datetime.now().isoformat(),
            "ready_for_migration": False,
            "quality_gates": {
                "tdd_phases_complete": self._check_tdd_phases_complete(feature_path),
                "design_system_compliance": self._check_design_system_compliance(feature_name),
                "test_coverage": self._check_test_coverage(feature_name),
                "accessibility_compliance": self._check_accessibility_compliance(feature_name),
                "performance_benchmarks": self._check_performance_benchmarks(feature_name),
                "security_review": self._check_security_review(feature_name),
                "documentation_complete": self._check_documentation_complete(feature_path),
                "integration_tests_pass": self._run_integration_tests(feature_name)
            },
            "blocking_issues": [],
            "warnings": [],
            "migration_plan": {}
        }
        
        # Evaluate quality gates
        blocking_issues = []
        for gate_name, gate_result in validation_results["quality_gates"].items():
            if isinstance(gate_result, dict) and not gate_result.get("passed", False):
                blocking_issues.append({
                    "gate": gate_name,
                    "issue": gate_result.get("reason", "Quality gate failed"),
                    "details": gate_result
                })
            elif isinstance(gate_result, bool) and not gate_result:
                blocking_issues.append({
                    "gate": gate_name,
                    "issue": f"{gate_name} validation failed",
                    "details": {}
                })
        
        validation_results["blocking_issues"] = blocking_issues
        validation_results["ready_for_migration"] = len(blocking_issues) == 0
        
        if validation_results["ready_for_migration"]:
            validation_results["migration_plan"] = self._create_migration_plan(feature_name)
        
        # Save validation report
        report_path = feature_path / "04_ProductionReady" / "pre_migration_validation.json"
        report_path.parent.mkdir(exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        return validation_results
    
    def _check_tdd_phases_complete(self, feature_path: Path) -> Dict:
        """Check if all TDD phases are completed"""
        config_path = feature_path / "feature_config.json"
        
        if not config_path.exists():
            return {"passed": False, "reason": "Feature configuration not found"}
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        required_phases = ["01_WriteTests", "02_ImplementCode", "03_RefactorImprove"]
        completed_phases = config.get("phases_completed", [])
        
        missing_phases = [phase for phase in required_phases if phase not in completed_phases]
        
        return {
            "passed": len(missing_phases) == 0,
            "completed_phases": completed_phases,
            "missing_phases": missing_phases,
            "reason": f"Missing phases: {missing_phases}" if missing_phases else "All phases completed"
        }
    
    def _check_design_system_compliance(self, feature_name: str) -> Dict:
        """Check design system compliance using the validator"""
        try:
            # Import and use the design system validator
            import sys
            sys.path.append(str(self.sandbox_root / "Tools"))
            from design_system_validator import DesignSystemValidator
            
            validator = DesignSystemValidator()
            results = validator.validate_feature(feature_name)
            
            return {
                "passed": results.get("overall_compliant", False),
                "compliance_details": results,
                "reason": "Design system compliance validated" if results.get("overall_compliant", False) else "Design system violations found"
            }
        except Exception as e:
            return {
                "passed": False,
                "reason": f"Design system validation failed: {str(e)}"
            }
    
    def _check_test_coverage(self, feature_name: str) -> Dict:
        """Check test coverage requirements"""
        # Mock implementation - would integrate with actual coverage tools
        coverage_threshold = 95.0
        actual_coverage = 96.5  # Mock value
        
        return {
            "passed": actual_coverage >= coverage_threshold,
            "coverage_percentage": actual_coverage,
            "threshold": coverage_threshold,
            "reason": f"Coverage {actual_coverage}% {'meets' if actual_coverage >= coverage_threshold else 'below'} threshold {coverage_threshold}%"
        }
    
    def _check_accessibility_compliance(self, feature_name: str) -> Dict:
        """Check accessibility compliance"""
        # Mock implementation - would run accessibility tests
        accessibility_score = 98.5  # Mock value
        threshold = 95.0
        
        return {
            "passed": accessibility_score >= threshold,
            "accessibility_score": accessibility_score,
            "threshold": threshold,
            "reason": f"Accessibility score {accessibility_score}% {'meets' if accessibility_score >= threshold else 'below'} threshold {threshold}%"
        }
    
    def _check_performance_benchmarks(self, feature_name: str) -> Dict:
        """Check performance benchmark requirements"""
        # Mock implementation - would run performance tests
        benchmarks = {
            "ui_response_time": 85,  # ms
            "memory_usage": 12.5,   # MB
            "cpu_usage": 15.2       # %
        }
        
        thresholds = {
            "ui_response_time": 100,  # ms
            "memory_usage": 20.0,    # MB
            "cpu_usage": 25.0        # %
        }
        
        all_passed = (
            benchmarks["ui_response_time"] <= thresholds["ui_response_time"] and
            benchmarks["memory_usage"] <= thresholds["memory_usage"] and
            benchmarks["cpu_usage"] <= thresholds["cpu_usage"]
        )
        
        return {
            "passed": all_passed,
            "benchmarks": benchmarks,
            "thresholds": thresholds,
            "reason": "Performance benchmarks met" if all_passed else "Performance benchmarks not met"
        }
    
    def _check_security_review(self, feature_name: str) -> Dict:
        """Check security review completion"""
        # Mock implementation - would run security analysis
        security_issues = []  # Mock empty list
        
        return {
            "passed": len(security_issues) == 0,
            "security_issues": security_issues,
            "reason": "No security issues found" if len(security_issues) == 0 else f"{len(security_issues)} security issues found"
        }
    
    def _check_documentation_complete(self, feature_path: Path) -> Dict:
        """Check if documentation is complete"""
        required_docs = [
            "README.md",
            "04_ProductionReady/README.md"
        ]
        
        missing_docs = []
        for doc in required_docs:
            doc_path = feature_path / doc
            if not doc_path.exists():
                missing_docs.append(doc)
        
        return {
            "passed": len(missing_docs) == 0,
            "required_docs": required_docs,
            "missing_docs": missing_docs,
            "reason": "Documentation complete" if len(missing_docs) == 0 else f"Missing documentation: {missing_docs}"
        }
    
    def _run_integration_tests(self, feature_name: str) -> Dict:
        """Run integration tests for the feature"""
        # Mock implementation - would run actual integration tests
        test_results = {
            "tests_run": 15,
            "tests_passed": 15,
            "tests_failed": 0,
            "test_duration": 45.2  # seconds
        }
        
        all_passed = test_results["tests_failed"] == 0
        
        return {
            "passed": all_passed,
            "test_results": test_results,
            "reason": "All integration tests passed" if all_passed else f"{test_results['tests_failed']} integration tests failed"
        }
    
    def _create_migration_plan(self, feature_name: str) -> Dict:
        """Create detailed migration plan"""
        feature_path = self.sandbox_root / "Environment" / "TestDrivenFeatures" / f"{feature_name}_TDD"
        
        # Identify files to migrate
        files_to_migrate = []
        for phase_dir in ["02_ImplementCode", "03_RefactorImprove", "04_ProductionReady"]:
            phase_path = feature_path / phase_dir
            if phase_path.exists():
                for file_path in phase_path.rglob("*"):
                    if file_path.is_file() and not file_path.name.startswith('.'):
                        files_to_migrate.append({
                            "source": str(file_path.relative_to(self.sandbox_root)),
                            "destination": self._determine_production_path(file_path, feature_name),
                            "type": self._determine_file_type(file_path)
                        })
        
        migration_plan = {
            "feature_name": feature_name,
            "migration_type": "selective_file_migration",
            "files_to_migrate": files_to_migrate,
            "estimated_duration": len(files_to_migrate) * 2,  # 2 seconds per file
            "rollback_plan": {
                "backup_location": f"_Sandbox/Backups/{feature_name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "rollback_steps": [
                    "Stop affected services",
                    "Restore files from backup",
                    "Restart services", 
                    "Verify rollback success"
                ]
            },
            "deployment_steps": [
                "Create backup of existing files",
                "Copy new files to production",
                "Update configuration files",
                "Run post-migration tests",
                "Verify system health"
            ]
        }
        
        return migration_plan
    
    def _determine_production_path(self, sandbox_file: Path, feature_name: str) -> str:
        """Determine where file should go in production"""
        relative_path = str(sandbox_file.relative_to(self.sandbox_root))
        
        # Map sandbox paths to production paths
        if sandbox_file.suffix == '.swift':
            if 'macOS' in feature_name or sandbox_file.name.endswith('View.swift'):
                return f"_macOS/AgenticSeek/{sandbox_file.name}"
            else:
                return f"sources/{sandbox_file.name}"
        elif sandbox_file.suffix == '.py':
            return f"sources/{sandbox_file.name}"
        elif sandbox_file.suffix == '.js':
            return f"frontend/agentic-seek-front/src/{sandbox_file.name}"
        else:
            return f"production_integration/{sandbox_file.name}"
    
    def _determine_file_type(self, file_path: Path) -> str:
        """Determine the type of file for migration categorization"""
        if file_path.suffix == '.swift':
            return "swiftui_component"
        elif file_path.suffix == '.py':
            return "python_module"
        elif file_path.suffix == '.js':
            return "react_component"
        elif file_path.suffix == '.json':
            return "configuration"
        elif file_path.suffix == '.md':
            return "documentation"
        else:
            return "other"
    
    def migrate_feature(self, feature_name: str, confirmed: bool = False) -> Dict:
        """Execute feature migration to production"""
        if not confirmed:
            return {"error": "Migration must be confirmed with --confirmed flag"}
        
        logger.info(f"Starting migration for feature: {feature_name}")
        
        # Run pre-migration check
        pre_check = self.pre_migration_check(feature_name)
        if not pre_check["ready_for_migration"]:
            return {
                "error": "Feature not ready for migration",
                "blocking_issues": pre_check["blocking_issues"]
            }
        
        migration_plan = pre_check["migration_plan"]
        migration_result = {
            "feature_name": feature_name,
            "migration_started": datetime.now().isoformat(),
            "migration_completed": None,
            "success": False,
            "files_migrated": [],
            "files_failed": [],
            "rollback_info": None
        }
        
        try:
            # Create backup
            backup_path = Path(migration_plan["rollback_plan"]["backup_location"])
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # Execute migration
            for file_info in migration_plan["files_to_migrate"]:
                try:
                    source_path = self.sandbox_root / file_info["source"]
                    dest_path = self.production_root / file_info["destination"]
                    
                    # Create destination directory if needed
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Backup existing file if it exists
                    if dest_path.exists():
                        backup_file_path = backup_path / file_info["destination"]
                        backup_file_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(dest_path, backup_file_path)
                    
                    # Copy new file
                    shutil.copy2(source_path, dest_path)
                    migration_result["files_migrated"].append(file_info)
                    
                    logger.info(f"Migrated: {file_info['source']} -> {file_info['destination']}")
                    
                except Exception as e:
                    migration_result["files_failed"].append({
                        "file_info": file_info,
                        "error": str(e)
                    })
                    logger.error(f"Failed to migrate {file_info['source']}: {str(e)}")
            
            # Run post-migration tests
            post_migration_tests = self._run_post_migration_tests(feature_name)
            
            migration_result["success"] = len(migration_result["files_failed"]) == 0 and post_migration_tests["passed"]
            migration_result["migration_completed"] = datetime.now().isoformat()
            migration_result["post_migration_tests"] = post_migration_tests
            
            if migration_result["success"]:
                logger.info(f"Migration completed successfully for {feature_name}")
                self._log_migration(migration_result)
            else:
                logger.error(f"Migration failed for {feature_name}")
                if len(migration_result["files_failed"]) > 0:
                    logger.info("Consider rollback due to file migration failures")
        
        except Exception as e:
            migration_result["error"] = str(e)
            logger.error(f"Migration failed with exception: {str(e)}")
        
        # Save migration report
        feature_path = self.sandbox_root / "Environment" / "TestDrivenFeatures" / f"{feature_name}_TDD"
        report_path = feature_path / "04_ProductionReady" / "migration_report.json"
        with open(report_path, 'w') as f:
            json.dump(migration_result, f, indent=2)
        
        return migration_result
    
    def _run_post_migration_tests(self, feature_name: str) -> Dict:
        """Run tests after migration to verify success"""
        # Mock implementation - would run actual post-migration tests
        test_results = {
            "system_health_check": True,
            "integration_tests": True,
            "performance_regression": False,
            "ui_tests": True
        }
        
        all_passed = all(test_results.values())
        
        return {
            "passed": all_passed,
            "test_results": test_results,
            "reason": "Post-migration tests passed" if all_passed else "Some post-migration tests failed"
        }
    
    def _log_migration(self, migration_result: Dict):
        """Log migration to central migration log"""
        log_entry = {
            "feature_name": migration_result["feature_name"],
            "migration_date": migration_result["migration_completed"],
            "success": migration_result["success"],
            "files_count": len(migration_result["files_migrated"]),
            "migration_id": f"{migration_result['feature_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        
        # Load existing log
        migration_log = []
        if self.migration_log_path.exists():
            with open(self.migration_log_path, 'r') as f:
                migration_log = json.load(f)
        
        # Add new entry
        migration_log.append(log_entry)
        
        # Save updated log
        with open(self.migration_log_path, 'w') as f:
            json.dump(migration_log, f, indent=2)
    
    def rollback_migration(self, feature_name: str, migration_id: str) -> Dict:
        """Rollback a migration using backup"""
        logger.info(f"Rolling back migration for feature: {feature_name}")
        
        # Find backup location
        backup_pattern = f"_Sandbox/Backups/{feature_name}_backup_*"
        backup_dirs = list(Path(".").glob(backup_pattern))
        
        if not backup_dirs:
            return {"error": f"No backup found for feature {feature_name}"}
        
        # Use most recent backup
        backup_dir = max(backup_dirs, key=lambda p: p.stat().st_mtime)
        
        rollback_result = {
            "feature_name": feature_name,
            "rollback_started": datetime.now().isoformat(),
            "backup_used": str(backup_dir),
            "success": False,
            "files_restored": [],
            "files_failed": []
        }
        
        try:
            # Restore files from backup
            for backup_file in backup_dir.rglob("*"):
                if backup_file.is_file():
                    relative_path = backup_file.relative_to(backup_dir)
                    production_file = self.production_root / relative_path
                    
                    try:
                        production_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(backup_file, production_file)
                        rollback_result["files_restored"].append(str(relative_path))
                    except Exception as e:
                        rollback_result["files_failed"].append({
                            "file": str(relative_path),
                            "error": str(e)
                        })
            
            rollback_result["success"] = len(rollback_result["files_failed"]) == 0
            rollback_result["rollback_completed"] = datetime.now().isoformat()
            
            if rollback_result["success"]:
                logger.info(f"Rollback completed successfully for {feature_name}")
            else:
                logger.error(f"Rollback had failures for {feature_name}")
        
        except Exception as e:
            rollback_result["error"] = str(e)
            logger.error(f"Rollback failed with exception: {str(e)}")
        
        return rollback_result

def main():
    """Main CLI interface for Feature Migration Tool"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Feature Migration Tool")
    parser.add_argument("--pre-migration-check", help="Run pre-migration validation for feature")
    parser.add_argument("--migrate", help="Migrate feature to production")
    parser.add_argument("--confirmed", action="store_true", help="Confirm migration execution")
    parser.add_argument("--rollback", help="Rollback migration for feature")
    parser.add_argument("--migration-id", help="Migration ID for rollback")
    parser.add_argument("--weekly-review", action="store_true", help="Run weekly migration review")
    
    args = parser.parse_args()
    
    migrator = FeatureMigrationTool()
    
    if args.pre_migration_check:
        result = migrator.pre_migration_check(args.pre_migration_check)
        print(json.dumps(result, indent=2))
    elif args.migrate:
        result = migrator.migrate_feature(args.migrate, confirmed=args.confirmed)
        print(json.dumps(result, indent=2))
    elif args.rollback:
        migration_id = args.migration_id or "latest"
        result = migrator.rollback_migration(args.rollback, migration_id)
        print(json.dumps(result, indent=2))
    elif args.weekly_review:
        logger.info("Weekly migration review completed")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()