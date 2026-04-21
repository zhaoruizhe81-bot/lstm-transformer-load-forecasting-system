import time
print("开始导入...")
t = time.time()
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'frontend'))
sys.path.insert(0, os.path.dirname(__file__))
from ui_utils.api_client import APIClient
print(f"导入耗时: {time.time()-t:.2f}s")

t = time.time()
api = APIClient()
r = api.get_all_users()
print(f"查询耗时: {time.time()-t:.2f}s")
print(f"结果: code={r['code']}, msg={r['message']}")
if r['code'] == 200:
    print(f"用户数: {r['data']['count']}")
else:
    print(f"错误: {r['message']}")

