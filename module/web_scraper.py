import requests
from bs4 import BeautifulSoup

def fetch_web_content_by_query(query):
    """
    通过搜索引擎查询获取网页内容。

    参数：
    query (str): 要搜索的查询字符串。

    返回：
    str: 获取的网页内容
    """
    search_url = f"https://www.google.com/search?q={query}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }

    try:
        response = requests.get(search_url, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        search_results = soup.find_all('div', class_='BNeawe s3v9rd AP7Wnd')

        content = "\n".join([result.get_text() for result in search_results])
        return content[:1000]
    except requests.RequestException as e:
        return f"请求错误: {e}"
    except Exception as e:
        return f"处理错误: {e}"
