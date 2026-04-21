# 数据导入脚本 - 将ETT-small数据集导入MySQL
# 使用方法: python import_data.py

import pandas as pd
import numpy as np
import pymysql
from datetime import datetime
import sys
import os

# 数据库配置
DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': '123456',
    'database': 'electric_load_forecasting_system_with_deep_learning_2026',
    'charset': 'utf8mb4'
}

# 数据文件路径 - 使用ETT-small数据集（ETTh1，约1.7万条逐小时数据）
DATA_FILE = 'ETT-small/ETTh1.csv'

# 导入配置
UPLOAD_USERID = 2  # 导入用户ID（analyst01）
BATCH_SIZE = 1000  # 批量插入条数


def generate_weather_from_features(row):
    """根据ETT-small特征列生成气象数据
    HUFL/HULL/MUFL/MULL/LUFL/LULL 为变压器油温相关特征，
    这里用它们模拟温度和湿度。
    """
    # 用HUFL和HULL的均值模拟温度（缩放到合理区间 -10~40℃）
    temp_raw = (row['HUFL'] + row['HULL']) / 2
    temp = temp_raw * 3.0 + 10.0  # 线性映射
    temp = max(-15, min(45, temp))

    # 用MUFL和MULL的均值模拟湿度（缩放到 20~95%）
    hum_raw = (row['MUFL'] + row['MULL']) / 2
    humidity = hum_raw * 15.0 + 50.0  # 线性映射
    humidity = max(20, min(95, humidity))

    return round(temp, 2), round(humidity, 2)


def is_holiday(dt):
    """判断是否为节假日（简化版）"""
    holidays = [
        (1, 1), (1, 2), (1, 3),        # 元旦
        (5, 1), (5, 2), (5, 3),        # 劳动节
        (10, 1), (10, 2), (10, 3),     # 国庆
        (10, 4), (10, 5), (10, 6), (10, 7),
        (12, 25),                       # 圣诞
    ]
    return 1 if (dt.month, dt.day) in holidays else 0


def import_data():
    """主导入函数"""
    print(f"[INFO] 读取数据文件: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    print(f"   原始列名: {list(df.columns)}")

    # ETTh1列: date, HUFL, HULL, MUFL, MULL, LUFL, LULL, OT
    # OT (Oil Temperature) 作为负荷预测目标值
    df['datetime'] = pd.to_datetime(df['date'])
    df['loadvalue'] = df['OT']
    df = df.sort_values('datetime').reset_index(drop=True)
    df = df.dropna()

    total = len(df)
    print(f"[INFO] 总记录数: {total:,} 条")
    print(f"[INFO] 时间范围: {df['datetime'].min()} ~ {df['datetime'].max()}")
    print(f"[INFO] 负荷范围: {df['loadvalue'].min():.2f} ~ {df['loadvalue'].max():.2f}")

    # 缩放负荷值到合理的电网负荷范围 (3000-10000 MW)
    load_min, load_max = df['loadvalue'].min(), df['loadvalue'].max()
    target_min, target_max = 3000, 10000
    df['loadvalue'] = (df['loadvalue'] - load_min) / (load_max - load_min) * (target_max - target_min) + target_min

    # 连接数据库
    print(f"\n[INFO] 连接数据库...")
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # 清空旧数据
    cursor.execute("DELETE FROM loaddata")
    conn.commit()
    print("[INFO] 已清空旧数据")

    # 批量插入
    sql = """INSERT INTO loaddata
             (recordtime, loadvalue, temperature, humidity, holiday, weekday, datasource, uploaduserid)
             VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"""

    print(f"\n[INFO] 开始导入 {total:,} 条数据（每批 {BATCH_SIZE} 条）...")
    batch = []
    imported = 0

    for idx, row in df.iterrows():
        dt = row['datetime']
        load = round(row['loadvalue'], 2)
        temp, hum = generate_weather_from_features(row)
        holiday = is_holiday(dt)
        weekday = dt.isoweekday()  # 1=周一, 7=周日

        batch.append((
            dt.strftime('%Y-%m-%d %H:%M:%S'),
            load, temp, hum, holiday, weekday,
            'ett_small_etth1', UPLOAD_USERID
        ))

        if len(batch) >= BATCH_SIZE:
            cursor.executemany(sql, batch)
            conn.commit()
            imported += len(batch)
            pct = imported / total * 100
            print(f"   [OK] 已导入 {imported:>7,}/{total:,} ({pct:5.1f}%)")
            batch = []

    # 插入剩余数据
    if batch:
        cursor.executemany(sql, batch)
        conn.commit()
        imported += len(batch)
        print(f"   [OK] 已导入 {imported:>7,}/{total:,} (100.0%)")

    # 验证
    cursor.execute("SELECT COUNT(*) FROM loaddata")
    count = cursor.fetchone()[0]
    cursor.execute("SELECT MIN(recordtime), MAX(recordtime) FROM loaddata")
    time_range = cursor.fetchone()

    print(f"\n{'='*50}")
    print(f"[DONE] 导入完成!")
    print(f"   数据库记录数: {count:,} 条")
    print(f"   时间范围: {time_range[0]} ~ {time_range[1]}")
    print(f"{'='*50}")

    cursor.close()
    conn.close()


if __name__ == '__main__':
    import_data()

