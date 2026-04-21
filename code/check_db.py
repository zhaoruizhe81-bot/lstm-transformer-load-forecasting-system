# 检查数据库状态
import pymysql, os
conn = pymysql.connect(host='localhost',user='root',password='123456',
                       database='electric_load_forecasting_system_with_deep_learning_2026')
cur = conn.cursor()

print('=== 训练记录 ===')
cur.execute('SELECT trainid, configid, traindata, trainstatus, modelpath FROM trainrecord')
for r in cur.fetchall():
    print(r)

print('\n=== 模型版本 ===')
cur.execute('SELECT versionid, trainid, versionnumber, isactive FROM modelversion')
for r in cur.fetchall():
    print(r)

print('\n=== 模型文件是否存在 ===')
cur.execute('SELECT modelpath FROM trainrecord')
for r in cur.fetchall():
    path = r[0]
    e = os.path.exists(path) if path else False
    tag = 'YES' if e else 'NO'
    print(f'  {path} -> {tag}')

print('\n=== 负荷数据统计 ===')
cur.execute('SELECT MIN(recordtime), MAX(recordtime), COUNT(*) FROM loaddata')
print(cur.fetchone())

print('\n=== 预测任务 ===')
cur.execute('SELECT taskid, taskname, versionid, taskstatus FROM predicttask')
for r in cur.fetchall():
    print(r)

print('\n=== 预测结果 ===')
cur.execute('SELECT COUNT(*) FROM predictresult')
print('预测结果条数:', cur.fetchone()[0])

conn.close()

