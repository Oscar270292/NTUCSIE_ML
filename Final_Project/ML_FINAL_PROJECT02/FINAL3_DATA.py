import pandas as pd
import requests

# 從網路下載CSV檔案
url = 'http://tcgmetro.blob.core.windows.net/stationod/%E8%87%BA%E5%8C%97%E6%8D%B7%E9%81%8B%E6%AF%8F%E6%97%A5%E5%88%86%E6%99%82%E5%90%84%E7%AB%99OD%E6%B5%81%E9%87%8F%E7%B5%B1%E8%A8%88%E8%B3%87%E6%96%99_202311.csv'  # 替換為您要下載的CSV檔案網址
response = requests.get(url)

# 檢查是否成功取得檔案
if response.status_code == 200:
    # 將資料轉換為DataFrame
    data = pd.read_csv(url)

    # 印出資料前幾筆觀測值
    print(data.head())

    # 進行資料處理操作，例如選擇特定欄位、篩選資料等
    filtered_data = data[(data['in'] > "科技大樓") & (data['out'] == '科技大樓')]

    # 印出處理後的資料前幾筆觀測值
    print(filtered_data.head())

else:
    print('無法取得資料。')
