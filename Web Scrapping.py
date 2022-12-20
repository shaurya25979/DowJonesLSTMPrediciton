import requests
from bs4 import BeautifulSoup
def dow(URL):  
# Website URL
    URL = 'https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average'
    page = requests.get(URL)
    # parse html content
    web = BeautifulSoup(page.text, 'html.parser')
    table = web.find('table', class_='wikitable')
    tbody = table.find('tbody')
    trs = tbody.find_all('tr')
    tickers = []
    for tr in trs:
        st = str(tr.find_all('a', class_ = 'external text'))
        if len(st)>3:
            tickers.extend([st.split(sep='>')[1].split(sep='<')[0]])
    return tickers