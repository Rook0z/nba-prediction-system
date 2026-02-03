"""
Setup Verification Script
Run this to verify your project is set up correctly
"""

import os
import sys


def check_file(filepath, description):
    """Check if a file exists"""
    if os.path.exists(filepath):
        print(f"✅ {description}")
        return True
    else:
        print(f"❌ {description} - MISSING: {filepath}")
        return False


def check_directory(dirpath, description):
    """Check if a directory exists"""
    if os.path.isdir(dirpath):
        print(f"✅ {description}")
        return True
    else:
        print(f"❌ {description} - MISSING: {dirpath}")
        return False


def check_imports():
    """Check if required packages can be imported"""
    print("\n" + "="*60)
    print("CHECKING PYTHON PACKAGES")
    print("="*60)
    
    packages = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('sklearn', 'scikit-learn'),
        ('xgboost', 'xgboost'),
        ('nba_api', 'nba_api'),
        ('yaml', 'pyyaml')
    ]
    
    all_installed = True
    for import_name, package_name in packages:
        try:
            __import__(import_name)
            print(f"✅ {package_name} installed")
        except ImportError:
            print(f"❌ {package_name} NOT installed")
            all_installed = False
    
    return all_installed


def check_project_structure():
    """Check if project structure is correct"""
    print("\n" + "="*60)
    print("CHECKING PROJECT STRUCTURE")
    print("="*60)
    
    checks = []
    
    # Configuration files
    print("\n📋 Configuration Files:")
    checks.append(check_file('config/config.yaml', 'config.yaml'))
    checks.append(check_file('requirements.txt', 'requirements.txt'))
    
    # Source code files
    print("\n🐍 Source Code:")
    checks.append(check_file('src/__init__.py', 'src/__init__.py'))
    checks.append(check_file('src/data/__init__.py', 'src/data/__init__.py'))
    checks.append(check_file('src/data/nba_collector.py', 'nba_collector.py'))
    checks.append(check_file('src/data/data_processor.py', 'data_processor.py'))
    checks.append(check_file('src/utils/__init__.py', 'src/utils/__init__.py'))
    checks.append(check_file('src/utils/constants.py', 'constants.py'))
    
    # Scripts
    print("\n📜 Scripts:")
    checks.append(check_file('scripts/collect_data.py', 'collect_data.py'))
    
    # Directories
    print("\n📁 Required Directories:")
    checks.append(check_directory('data/raw/game_logs', 'data/raw/game_logs/'))
    checks.append(check_directory('data/processed', 'data/processed/'))
    checks.append(check_directory('models', 'models/'))
    checks.append(check_directory('results', 'results/'))
    
    return all(checks)


def test_imports():
    """Test if custom modules can be imported"""
    print("\n" + "="*60)
    print("TESTING CUSTOM MODULES")
    print("="*60)
    
    try:
        from src.data.nba_collector import NBADataCollector
        print("✅ Can import NBADataCollector")
    except Exception as e:
        print(f"❌ Cannot import NBADataCollector: {e}")
        return False
    
    try:
        from src.data.data_processor import NBADataProcessor
        print("✅ Can import NBADataProcessor")
    except Exception as e:
        print(f"❌ Cannot import NBADataProcessor: {e}")
        return False
    
    try:
        from src.utils import constants
        print("✅ Can import constants")
    except Exception as e:
        print(f"❌ Cannot import constants: {e}")
        return False
    
    return True


def main():
    """Main verification function"""
    
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║          NBA PREDICTION SYSTEM - VERIFICATION            ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Check structure
    structure_ok = check_project_structure()
    
    # Check packages
    packages_ok = check_imports()
    
    # Test custom imports
    imports_ok = test_imports()
    
    # Final summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    if structure_ok and packages_ok and imports_ok:
        print("\n🎉 ALL CHECKS PASSED!")
        print("\nYou're ready to collect data:")
        print("  python scripts/collect_data.py")
        return 0
    else:
        print("\n⚠️  SOME CHECKS FAILED")
        print("\nPlease fix the issues above before proceeding.")
        
        if not structure_ok:
            print("\n📋 Structure issues:")
            print("  - Make sure all files are in correct locations")
            print("  - Check FILE_CHECKLIST.md for reference")
        
        if not packages_ok:
            print("\n📦 Package issues:")
            print("  - Run: pip install -r requirements.txt")
        
        if not imports_ok:
            print("\n🐍 Import issues:")
            print("  - Verify __init__.py files exist")
            print("  - Check file syntax for errors")
        
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)