#!/usr/bin/env python3
"""
BIRD数据集数据库下载脚本
下载BIRD数据集所需的SQLite数据库文件
"""

import os
import sys
import json
import requests
import zipfile
import argparse
from pathlib import Path
from tqdm import tqdm
import hashlib


def calculate_file_hash(filepath):
    """计算文件的MD5哈希值"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def download_file(url, filepath, expected_hash=None):
    """下载文件并验证哈希值"""
    try:
        print(f"正在下载: {url}")
        
        # 发送请求
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # 获取文件大小
        total_size = int(response.headers.get('content-length', 0))
        
        # 创建父目录
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # 下载文件
        with open(filepath, 'wb') as f, tqdm(
            desc=f"下载 {filepath.name}",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        # 验证哈希值
        if expected_hash:
            actual_hash = calculate_file_hash(filepath)
            if actual_hash != expected_hash:
                print(f"❌ 哈希值验证失败: 期望 {expected_hash}, 实际 {actual_hash}")
                filepath.unlink()
                return False
            else:
                print(f"✓ 哈希值验证通过")
        
        return True
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        if filepath.exists():
            filepath.unlink()
        return False


def extract_zip(zip_path, extract_dir):
    """解压ZIP文件"""
    try:
        print(f"正在解压: {zip_path}")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # 获取ZIP文件中的文件列表
            file_list = zip_ref.namelist()
            
            # 解压文件
            for file in tqdm(file_list, desc="解压进度"):
                zip_ref.extract(file, extract_dir)
        
        print(f"✓ 解压完成: {extract_dir}")
        return True
        
    except Exception as e:
        print(f"❌ 解压失败: {e}")
        return False


def get_required_databases(data_dir):
    """获取需要的数据库列表"""
    dev_json_path = Path(data_dir) / "dev.json"
    
    if not dev_json_path.exists():
        print(f"❌ 数据集文件不存在: {dev_json_path}")
        return set()
    
    with open(dev_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取所有数据库ID
    db_ids = set()
    for item in data:
        db_ids.add(item['db_id'])
    
    return db_ids


def check_existing_databases(db_root, required_dbs):
    """检查已存在的数据库文件"""
    existing_dbs = set()
    missing_dbs = set()
    
    for db_id in required_dbs:
        db_path = Path(db_root) / db_id / f"{db_id}.sqlite"
        if db_path.exists():
            existing_dbs.add(db_id)
            print(f"✓ 数据库已存在: {db_id}")
        else:
            missing_dbs.add(db_id)
            print(f"❌ 数据库缺失: {db_id}")
    
    return existing_dbs, missing_dbs


def download_bird_databases(data_dir="LPE-SQL/data", db_root="LPE-SQL/data"):
    """下载BIRD数据集数据库"""
    print("=== BIRD数据集数据库下载器 ===")
    
    data_dir = Path(data_dir)
    db_root = Path(db_root)
    
    # 检查数据集文件
    if not (data_dir / "dev.json").exists():
        print(f"❌ 数据集文件不存在: {data_dir / 'dev.json'}")
        print("请确保BIRD数据集文件已正确放置")
        return False
    
    # 获取需要的数据库
    required_dbs = get_required_databases(data_dir)
    print(f"需要 {len(required_dbs)} 个数据库")
    
    # 检查已存在的数据库
    existing_dbs, missing_dbs = check_existing_databases(db_root, required_dbs)
    
    if not missing_dbs:
        print("✓ 所有数据库文件都已存在")
        return True
    
    print(f"需要下载 {len(missing_dbs)} 个数据库")
    
    # BIRD数据集官方下载链接
    # 注意：这里需要根据实际的BIRD数据集下载链接进行配置
    bird_download_urls = {
        # 示例链接，需要替换为实际的下载链接
        # "california_schools": "https://example.com/bird/california_schools.zip",
        # "student_math": "https://example.com/bird/student_math.zip",
    }
    
    # 如果没有配置下载链接，提供手动下载指导
    if not bird_download_urls:
        print("\n⚠️  自动下载链接未配置")
        print("请手动下载BIRD数据集数据库文件：")
        print("1. 访问 BIRD 数据集官方页面")
        print("2. 下载所有数据库文件")
        print("3. 将数据库文件解压到以下目录：")
        print(f"   {db_root}")
        print("\n需要的数据库文件：")
        for db_id in sorted(missing_dbs):
            print(f"  - {db_id}/{db_id}.sqlite")
        
        return False
    
    # 下载缺失的数据库
    success_count = 0
    for db_id in missing_dbs:
        if db_id not in bird_download_urls:
            print(f"⚠️  没有找到 {db_id} 的下载链接")
            continue
        
        download_url = bird_download_urls[db_id]
        zip_path = data_dir / f"{db_id}.zip"
        
        # 下载ZIP文件
        if download_file(download_url, zip_path):
            # 解压文件
            extract_dir = db_root / db_id
            if extract_zip(zip_path, extract_dir):
                # 验证数据库文件
                db_file = extract_dir / f"{db_id}.sqlite"
                if db_file.exists():
                    print(f"✓ 数据库下载成功: {db_id}")
                    success_count += 1
                else:
                    print(f"❌ 解压后未找到数据库文件: {db_id}")
                
                # 删除ZIP文件
                zip_path.unlink()
        
    print(f"\n下载完成: {success_count}/{len(missing_dbs)} 个数据库")
    
    return success_count == len(missing_dbs)


def create_sample_database():
    """创建示例数据库用于测试"""
    print("创建示例数据库...")
    
    import sqlite3
    
    # 创建示例数据库目录
    sample_db_dir = Path("LPE-SQL/data/california_schools")
    sample_db_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建示例数据库
    sample_db_path = sample_db_dir / "california_schools.sqlite"
    
    if sample_db_path.exists():
        print("✓ 示例数据库已存在")
        return True
    
    try:
        conn = sqlite3.connect(sample_db_path)
        cursor = conn.cursor()
        
        # 创建示例表
        cursor.execute('''
            CREATE TABLE frpm (
                "CDSCode" TEXT,
                "County Name" TEXT,
                "District Name" TEXT,
                "School Name" TEXT,
                "Enrollment (K-12)" INTEGER,
                "Free Meal Count (K-12)" INTEGER,
                "Educational Option Type" TEXT,
                "Charter School (Y/N)" INTEGER,
                "Charter Funding Type" TEXT
            )
        ''')
        
        # 插入示例数据
        sample_data = [
            ("123456789", "Alameda", "Alameda Unified", "Alameda High School", 1000, 300, "Regular", 0, "Not Applicable"),
            ("123456790", "Alameda", "Alameda Unified", "Alameda Middle School", 800, 200, "Regular", 0, "Not Applicable"),
            ("123456791", "Alameda", "Alameda Unified", "Alameda Elementary School", 600, 150, "Regular", 0, "Not Applicable"),
        ]
        
        cursor.executemany('''
            INSERT INTO frpm VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', sample_data)
        
        conn.commit()
        conn.close()
        
        print(f"✓ 示例数据库创建成功: {sample_db_path}")
        return True
        
    except Exception as e:
        print(f"❌ 创建示例数据库失败: {e}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="BIRD数据集数据库下载器")
    parser.add_argument("--data-dir", type=str, default="LPE-SQL/data",
                       help="BIRD数据集目录")
    parser.add_argument("--db-root", type=str, default="LPE-SQL/data",
                       help="数据库根目录")
    parser.add_argument("--create-sample", action="store_true",
                       help="创建示例数据库用于测试")
    
    args = parser.parse_args()
    
    try:
        if args.create_sample:
            success = create_sample_database()
        else:
            success = download_bird_databases(args.data_dir, args.db_root)
        
        if success:
            print("\n✓ 数据库下载/创建完成")
            print("现在可以运行BIRD数据集评估：")
            print("  python run_example.py --bird")
        else:
            print("\n❌ 数据库下载/创建失败")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()