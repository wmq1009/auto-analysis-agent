#!/usr/bin/env python3
"""
测试自动安装模块功能
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.auto_install import install_missing_modules, ensure_matplotlib_pyplot

def test_ensure_matplotlib_pyplot():
    """测试确保matplotlib.pyplot可以正常导入"""
    print("测试ensure_matplotlib_pyplot...")
    success = ensure_matplotlib_pyplot()
    if success:
        print("✓ matplotlib.pyplot可以正常导入")
        # 尝试直接导入，确认功能正常
        import matplotlib.pyplot as plt
        print("✓ 成功导入matplotlib.pyplot")
        plt.close()
    else:
        print("✗ matplotlib.pyplot导入失败")
    return success

def test_install_missing_modules():
    """测试自动安装缺失的模块"""
    print("\n测试install_missing_modules...")
    # 测试安装一个不太常用但可用的模块
    test_modules = ["pyparsing"]  # pyparsing是matplotlib的依赖，应该已经安装
    installed_modules = install_missing_modules(test_modules)
    if installed_modules:
        print(f"✓ 成功安装/检测到模块: {installed_modules}")
    else:
        print("✗ 模块安装失败")
    return len(installed_modules) > 0

def test_direct_import():
    """测试直接导入matplotlib.pyplot"""
    print("\n测试直接导入matplotlib.pyplot...")
    try:
        import matplotlib.pyplot as plt
        print("✓ 成功直接导入matplotlib.pyplot")
        plt.close()
        return True
    except Exception as e:
        print(f"✗ 直接导入失败: {e}")
        return False

if __name__ == "__main__":
    print("=== 自动安装模块功能测试 ===")
    
    test_results = []
    test_results.append(test_ensure_matplotlib_pyplot())
    test_results.append(test_install_missing_modules())
    test_results.append(test_direct_import())
    
    print("\n=== 测试结果 ===")
    if all(test_results):
        print("✓ 所有测试通过！自动安装模块功能正常工作")
    else:
        failed_tests = [i+1 for i, result in enumerate(test_results) if not result]
        print(f"✗ 测试失败: {failed_tests}")
        sys.exit(1)
    
    sys.exit(0)