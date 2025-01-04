import pandas as pd

# 替换为你的 feather 文件路径
file_path = "data/binance/BTC_USDT-1h.feather"

# 加载数据
data = pd.read_feather(file_path)

# 显示前几行数据
print(data.head())

# 显示列信息
print(data.info())
