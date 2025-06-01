#!/usr/bin/env python3
"""
Sandbox Cleanup Automation
Manages sandbox environment maintenance, cleanup, and optimization
Ensures efficient use of development resources
"""

import os
import shutil
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SandboxCleanupAutomation:
    """
    Automates sandbox environment cleanup and maintenance
    """
    
    def __init__(self, sandbox_root: str = "_Sandbox"):
        self.sandbox_root = Path(sandbox_root)
        self.cleanup_config_path = self.sandbox_root / "cleanup_config.json"
        self.cleanup_log_path = self.sandbox_root / "cleanup_log.json"
        
        # Load or create cleanup configuration
        self.config = self._load_or_create_config()
    
    def _load_or_create_config(self) -> Dict:
        """Load cleanup configuration or create default"""
        default_config = {
            "retention_policies": {
                "completed_features": 30,  # days
                "failed_features": 7,      # days
                "test_artifacts": 3,       # days
                "performance_reports": 14, # days
                "migration_backups": 90    # days
            },
            "cleanup_schedules": {
                "daily": ["temp_files", "test_artifacts"],
                "weekly": ["completed_features", "performance_reports"],
                "monthly": ["migration_backups", "archived_features"]
            },
            "size_limits": {
                "max_feature_size_mb": 100,
                "max_total_sandbox_size_gb": 5,
                "max_backup_size_gb": 2
            },
            "auto_archive": {
                "enabled": True,
                "archive_after_days": 30,
                "compression_enabled": True
            }
        }
        
        if self.cleanup_config_path.exists():
            with open(self.cleanup_config_path, 'r') as f:
                config = json.load(f)
            # Merge with defaults for any missing keys
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
        else:
            config = default_config
            self._save_config(config)
        
        return config
    
    def _save_config(self, config: Dict):
        """Save cleanup configuration"""
        with open(self.cleanup_config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def daily_maintenance(self) -> Dict:
        """Perform daily sandbox maintenance"""
        logger.info("Starting daily sandbox maintenance")
        
        maintenance_result = {
            "timestamp": datetime.now().isoformat(),
            "maintenance_type": "daily",
            "actions_performed": [],
            "space_freed_mb": 0,
            "errors": []
        }
        
        try:
            # Clean temporary files
            temp_cleaned = self._clean_temp_files()
            maintenance_result["actions_performed"].append("temp_files_cleaned")
            maintenance_result["space_freed_mb"] += temp_cleaned["space_freed_mb"]
            
            # Clean test artifacts
            artifacts_cleaned = self._clean_test_artifacts()
            maintenance_result["actions_performed"].append("test_artifacts_cleaned")
            maintenance_result["space_freed_mb"] += artifacts_cleaned["space_freed_mb"]
            
            # Check disk usage
            disk_usage = self._check_disk_usage()
            maintenance_result["disk_usage"] = disk_usage
            
            # Optimize sandbox structure
            optimization_result = self._optimize_sandbox_structure()
            maintenance_result["actions_performed"].append("structure_optimized")
            
            logger.info(f"Daily maintenance completed. Space freed: {maintenance_result['space_freed_mb']:.1f}MB")
            
        except Exception as e:
            maintenance_result["errors"].append(str(e))
            logger.error(f"Daily maintenance error: {str(e)}")
        
        self._log_cleanup_action(maintenance_result)
        return maintenance_result
    
    def weekly_maintenance(self) -> Dict:
        """Perform weekly sandbox maintenance"""
        logger.info("Starting weekly sandbox maintenance")
        
        maintenance_result = {
            "timestamp": datetime.now().isoformat(),
            "maintenance_type": "weekly",
            "actions_performed": [],
            "space_freed_mb": 0,
            "errors": []
        }
        
        try:
            # Archive completed features
            archive_result = self._archive_completed_features()
            maintenance_result["actions_performed"].append("features_archived")
            maintenance_result["space_freed_mb"] += archive_result["space_freed_mb"]
            
            # Clean performance reports
            reports_cleaned = self._clean_performance_reports()
            maintenance_result["actions_performed"].append("performance_reports_cleaned")
            maintenance_result["space_freed_mb"] += reports_cleaned["space_freed_mb"]
            
            # Validate sandbox integrity
            integrity_check = self._validate_sandbox_integrity()
            maintenance_result["integrity_check"] = integrity_check
            
            # Generate usage statistics
            usage_stats = self._generate_usage_statistics()
            maintenance_result["usage_statistics"] = usage_stats
            
            logger.info(f"Weekly maintenance completed. Space freed: {maintenance_result['space_freed_mb']:.1f}MB")
            
        except Exception as e:
            maintenance_result["errors"].append(str(e))
            logger.error(f"Weekly maintenance error: {str(e)}")
        
        self._log_cleanup_action(maintenance_result)
        return maintenance_result
    
    def _clean_temp_files(self) -> Dict:
        """Clean temporary files from sandbox"""
        logger.info("Cleaning temporary files")
        
        temp_patterns = [
            "**/*.tmp",
            "**/*.temp", 
            "**/.DS_Store",
            "**/Thumbs.db",
            "**/__pycache__",
            "**/*.pyc",
            "**/.pytest_cache",
            "**/node_modules"
        ]
        
        space_freed = 0
        files_removed = 0
        
        for pattern in temp_patterns:
            for temp_file in self.sandbox_root.glob(pattern):
                try:
                    if temp_file.is_file():
                        space_freed += temp_file.stat().st_size
                        temp_file.unlink()
                        files_removed += 1
                    elif temp_file.is_dir():
                        space_freed += self._get_directory_size(temp_file)
                        shutil.rmtree(temp_file)
                        files_removed += 1
                except Exception as e:
                    logger.warning(f"Could not remove {temp_file}: {str(e)}")
        
        return {
            "space_freed_mb": space_freed / (1024 * 1024),
            "files_removed": files_removed
        }
    
    def _clean_test_artifacts(self) -> Dict:
        """Clean old test artifacts"""
        logger.info("Cleaning test artifacts")
        
        cutoff_date = datetime.now() - timedelta(days=self.config["retention_policies"]["test_artifacts"])
        space_freed = 0
        files_removed = 0
        
        # Find test artifact patterns
        artifact_patterns = [
            "**/test_results_*.json",
            "**/coverage_*.html",
            "**/performance_test_*.log",
            "**/screenshot_*.png",
            "**/test_output_*.txt"
        ]
        
        for pattern in artifact_patterns:
            for artifact_file in self.sandbox_root.glob(pattern):
                try:
                    file_mtime = datetime.fromtimestamp(artifact_file.stat().st_mtime)
                    if file_mtime < cutoff_date:
                        space_freed += artifact_file.stat().st_size
                        artifact_file.unlink()
                        files_removed += 1
                except Exception as e:
                    logger.warning(f"Could not remove {artifact_file}: {str(e)}")
        
        return {
            "space_freed_mb": space_freed / (1024 * 1024),
            "files_removed": files_removed
        }
    
    def _archive_completed_features(self) -> Dict:
        """Archive completed features older than retention policy"""
        logger.info("Archiving completed features")
        
        cutoff_date = datetime.now() - timedelta(days=self.config["retention_policies"]["completed_features"])
        features_path = self.sandbox_root / "Environment" / "TestDrivenFeatures"
        archive_path = self.sandbox_root / "Archives"
        archive_path.mkdir(exist_ok=True)
        
        space_freed = 0
        features_archived = 0
        
        if features_path.exists():
            for feature_dir in features_path.iterdir():
                if feature_dir.is_dir() and feature_dir.name.endswith("_TDD"):
                    try:
                        # Check if feature is completed and old enough
                        config_file = feature_dir / "feature_config.json"
                        if config_file.exists():
                            with open(config_file, 'r') as f:
                                config = json.load(f)
                            
                            created_date = datetime.fromisoformat(config.get("created_at", datetime.now().isoformat()))
                            phases_completed = config.get("phases_completed", [])
                            
                            # Archive if completed and old enough
                            if ("04_ProductionReady" in phases_completed or 
                                len(phases_completed) >= 3) and created_date < cutoff_date:
                                
                                # Calculate size before archiving
                                feature_size = self._get_directory_size(feature_dir)
                                
                                # Create archive
                                if self.config["auto_archive"]["compression_enabled"]:
                                    archive_file = archive_path / f"{feature_dir.name}_{datetime.now().strftime('%Y%m%d')}.tar.gz"
                                    self._create_compressed_archive(feature_dir, archive_file)
                                else:
                                    archive_dir = archive_path / feature_dir.name
                                    shutil.move(str(feature_dir), str(archive_dir))
                                
                                # Remove original
                                if feature_dir.exists():
                                    shutil.rmtree(feature_dir)
                                
                                space_freed += feature_size
                                features_archived += 1
                                
                                logger.info(f"Archived feature: {feature_dir.name}")
                    
                    except Exception as e:
                        logger.warning(f"Could not archive {feature_dir.name}: {str(e)}")
        
        return {
            "space_freed_mb": space_freed / (1024 * 1024),
            "features_archived": features_archived
        }
    
    def _clean_performance_reports(self) -> Dict:
        """Clean old performance reports"""
        logger.info("Cleaning performance reports")
        
        cutoff_date = datetime.now() - timedelta(days=self.config["retention_policies"]["performance_reports"])
        space_freed = 0
        reports_removed = 0
        
        performance_patterns = [
            "**/performance_report_*.json",
            "**/benchmark_results_*.json",
            "**/profiling_data_*.txt"
        ]
        
        for pattern in performance_patterns:
            for report_file in self.sandbox_root.glob(pattern):
                try:
                    file_mtime = datetime.fromtimestamp(report_file.stat().st_mtime)
                    if file_mtime < cutoff_date:
                        space_freed += report_file.stat().st_size
                        report_file.unlink()
                        reports_removed += 1
                except Exception as e:
                    logger.warning(f"Could not remove {report_file}: {str(e)}")
        
        return {
            "space_freed_mb": space_freed / (1024 * 1024),
            "reports_removed": reports_removed
        }
    
    def _check_disk_usage(self) -> Dict:
        """Check sandbox disk usage against limits"""
        sandbox_size = self._get_directory_size(self.sandbox_root)
        sandbox_size_gb = sandbox_size / (1024 * 1024 * 1024)
        
        usage_info = {
            "sandbox_size_gb": sandbox_size_gb,
            "max_size_gb": self.config["size_limits"]["max_total_sandbox_size_gb"],
            "usage_percentage": (sandbox_size_gb / self.config["size_limits"]["max_total_sandbox_size_gb"]) * 100,
            "over_limit": sandbox_size_gb > self.config["size_limits"]["max_total_sandbox_size_gb"]
        }
        
        if usage_info["over_limit"]:
            logger.warning(f"Sandbox size ({sandbox_size_gb:.1f}GB) exceeds limit ({self.config['size_limits']['max_total_sandbox_size_gb']}GB)")
        
        return usage_info
    
    def _optimize_sandbox_structure(self) -> Dict:
        """Optimize sandbox directory structure"""
        logger.info("Optimizing sandbox structure")
        
        optimizations = []
        
        # Ensure all required directories exist
        required_dirs = [
            "Environment/TestDrivenFeatures",
            "Environment/DesignSystemValidation",
            "Environment/UserExperienceLab",
            "Environment/IntegrationStaging",
            "Tools",
            "Archives",
            "Backups"
        ]
        
        for dir_path in required_dirs:
            full_path = self.sandbox_root / dir_path
            if not full_path.exists():
                full_path.mkdir(parents=True)
                optimizations.append(f"Created missing directory: {dir_path}")
        
        # Remove empty directories
        for root, dirs, files in os.walk(self.sandbox_root, topdown=False):
            for dir_name in dirs:
                dir_path = Path(root) / dir_name
                try:
                    if not any(dir_path.iterdir()):
                        dir_path.rmdir()
                        optimizations.append(f"Removed empty directory: {dir_path.relative_to(self.sandbox_root)}")
                except OSError:
                    pass  # Directory not empty or permission issue
        
        return {
            "optimizations_performed": optimizations,
            "optimization_count": len(optimizations)
        }
    
    def _validate_sandbox_integrity(self) -> Dict:
        """Validate sandbox environment integrity"""
        logger.info("Validating sandbox integrity")
        
        integrity_issues = []
        
        # Check for required files
        required_files = [
            "Tools/sandbox_tdd_runner.py",
            "Tools/design_system_validator.py",
            "Tools/feature_migration_tool.py"
        ]
        
        for file_path in required_files:
            full_path = self.sandbox_root / file_path
            if not full_path.exists():
                integrity_issues.append(f"Missing required file: {file_path}")
        
        # Check for corrupted feature configs
        features_path = self.sandbox_root / "Environment" / "TestDrivenFeatures"
        if features_path.exists():
            for feature_dir in features_path.iterdir():
                if feature_dir.is_dir() and feature_dir.name.endswith("_TDD"):
                    config_file = feature_dir / "feature_config.json"
                    if config_file.exists():
                        try:
                            with open(config_file, 'r') as f:
                                json.load(f)
                        except json.JSONDecodeError:
                            integrity_issues.append(f"Corrupted config file: {config_file.relative_to(self.sandbox_root)}")
        
        return {
            "integrity_valid": len(integrity_issues) == 0,
            "issues_found": integrity_issues,
            "issues_count": len(integrity_issues)
        }
    
    def _generate_usage_statistics(self) -> Dict:
        """Generate sandbox usage statistics"""
        logger.info("Generating usage statistics")
        
        features_path = self.sandbox_root / "Environment" / "TestDrivenFeatures"
        
        stats = {
            "total_features": 0,
            "active_features": 0,
            "completed_features": 0,
            "archived_features": 0,
            "average_feature_size_mb": 0,
            "total_size_gb": self._get_directory_size(self.sandbox_root) / (1024 * 1024 * 1024),
            "most_recent_activity": None
        }
        
        if features_path.exists():
            feature_sizes = []
            most_recent = None
            
            for feature_dir in features_path.iterdir():
                if feature_dir.is_dir() and feature_dir.name.endswith("_TDD"):
                    stats["total_features"] += 1
                    
                    # Get feature size
                    feature_size = self._get_directory_size(feature_dir)
                    feature_sizes.append(feature_size)
                    
                    # Check feature status
                    config_file = feature_dir / "feature_config.json"
                    if config_file.exists():
                        try:
                            with open(config_file, 'r') as f:
                                config = json.load(f)
                            
                            phases_completed = config.get("phases_completed", [])
                            if "04_ProductionReady" in phases_completed:
                                stats["completed_features"] += 1
                            else:
                                stats["active_features"] += 1
                            
                            # Track most recent activity
                            created_date = datetime.fromisoformat(config.get("created_at", datetime.now().isoformat()))
                            if most_recent is None or created_date > most_recent:
                                most_recent = created_date
                        
                        except (json.JSONDecodeError, ValueError):
                            pass  # Skip corrupted configs
            
            if feature_sizes:
                stats["average_feature_size_mb"] = sum(feature_sizes) / len(feature_sizes) / (1024 * 1024)
            
            if most_recent:
                stats["most_recent_activity"] = most_recent.isoformat()
        
        # Check archives
        archive_path = self.sandbox_root / "Archives"
        if archive_path.exists():
            stats["archived_features"] = len([f for f in archive_path.iterdir() if f.is_file() or f.is_dir()])
        
        return stats
    
    def _get_directory_size(self, directory: Path) -> int:
        """Get total size of directory in bytes"""
        total_size = 0
        try:
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except (OSError, PermissionError):
            pass  # Skip inaccessible files
        return total_size
    
    def _create_compressed_archive(self, source_dir: Path, archive_file: Path):
        """Create compressed archive of directory"""
        import tarfile
        
        with tarfile.open(archive_file, "w:gz") as tar:
            tar.add(source_dir, arcname=source_dir.name)
    
    def _log_cleanup_action(self, action_result: Dict):
        """Log cleanup action to central log"""
        # Load existing log
        cleanup_log = []
        if self.cleanup_log_path.exists():
            try:
                with open(self.cleanup_log_path, 'r') as f:
                    cleanup_log = json.load(f)
            except json.JSONDecodeError:
                cleanup_log = []
        
        # Add new entry
        cleanup_log.append(action_result)
        
        # Keep only last 100 entries
        cleanup_log = cleanup_log[-100:]
        
        # Save updated log
        with open(self.cleanup_log_path, 'w') as f:
            json.dump(cleanup_log, f, indent=2)
    
    def emergency_cleanup(self) -> Dict:
        """Perform emergency cleanup when space is critically low"""
        logger.warning("Performing emergency cleanup")
        
        cleanup_result = {
            "timestamp": datetime.now().isoformat(),
            "cleanup_type": "emergency",
            "space_freed_mb": 0,
            "actions_performed": []
        }
        
        # Aggressive cleanup actions
        actions = [
            ("temp_files", self._clean_temp_files),
            ("test_artifacts", self._clean_test_artifacts),
            ("performance_reports", self._clean_performance_reports),
            ("force_archive_old_features", self._force_archive_old_features)
        ]
        
        for action_name, action_func in actions:
            try:
                result = action_func()
                cleanup_result["space_freed_mb"] += result.get("space_freed_mb", 0)
                cleanup_result["actions_performed"].append(action_name)
            except Exception as e:
                logger.error(f"Emergency cleanup action {action_name} failed: {str(e)}")
        
        self._log_cleanup_action(cleanup_result)
        return cleanup_result
    
    def _force_archive_old_features(self) -> Dict:
        """Force archive features older than 7 days (emergency)"""
        cutoff_date = datetime.now() - timedelta(days=7)  # More aggressive than normal
        return self._archive_features_older_than(cutoff_date)
    
    def _archive_features_older_than(self, cutoff_date: datetime) -> Dict:
        """Archive features older than specified date"""
        features_path = self.sandbox_root / "Environment" / "TestDrivenFeatures"
        archive_path = self.sandbox_root / "Archives"
        archive_path.mkdir(exist_ok=True)
        
        space_freed = 0
        features_archived = 0
        
        if features_path.exists():
            for feature_dir in features_path.iterdir():
                if feature_dir.is_dir() and feature_dir.name.endswith("_TDD"):
                    try:
                        dir_mtime = datetime.fromtimestamp(feature_dir.stat().st_mtime)
                        if dir_mtime < cutoff_date:
                            feature_size = self._get_directory_size(feature_dir)
                            
                            # Create archive
                            archive_file = archive_path / f"{feature_dir.name}_{datetime.now().strftime('%Y%m%d')}.tar.gz"
                            self._create_compressed_archive(feature_dir, archive_file)
                            
                            # Remove original
                            shutil.rmtree(feature_dir)
                            
                            space_freed += feature_size
                            features_archived += 1
                    
                    except Exception as e:
                        logger.warning(f"Could not archive {feature_dir.name}: {str(e)}")
        
        return {
            "space_freed_mb": space_freed / (1024 * 1024),
            "features_archived": features_archived
        }

def main():
    """Main CLI interface for Sandbox Cleanup Automation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Sandbox Cleanup Automation")
    parser.add_argument("--daily-maintenance", action="store_true", help="Run daily maintenance")
    parser.add_argument("--weekly-maintenance", action="store_true", help="Run weekly maintenance")
    parser.add_argument("--emergency-cleanup", action="store_true", help="Run emergency cleanup")
    parser.add_argument("--usage-stats", action="store_true", help="Generate usage statistics")
    parser.add_argument("--integrity-check", action="store_true", help="Check sandbox integrity")
    
    args = parser.parse_args()
    
    cleanup = SandboxCleanupAutomation()
    
    if args.daily_maintenance:
        result = cleanup.daily_maintenance()
        print(json.dumps(result, indent=2))
    elif args.weekly_maintenance:
        result = cleanup.weekly_maintenance()
        print(json.dumps(result, indent=2))
    elif args.emergency_cleanup:
        result = cleanup.emergency_cleanup()
        print(json.dumps(result, indent=2))
    elif args.usage_stats:
        stats = cleanup._generate_usage_statistics()
        print(json.dumps(stats, indent=2))
    elif args.integrity_check:
        integrity = cleanup._validate_sandbox_integrity()
        print(json.dumps(integrity, indent=2))
    else:
        parser.print_help()

if __name__ == "__main__":
    main()