#!/usr/bin/env python
# coding: utf-8

# In[117]:


from bs4 import BeautifulSoup as soup
import requests


# In[92]:


from datetime import datetime as dt
date = ['10/09/2023', '10/08/2023']
for i in range(len(date)):
    date[i] = dt.strptime(date[i], '%m/%d/%Y').date()   


# In[201]:


import re

cnn_url = "https://edition.cnn.com/weather" 

html = requests.get(cnn_url).content
html = soup(html, 'html.parser')

def get_href(link):
    return link and re.compile("index.html$").search(link)

for link in html.findAll("a", href=get_href):
    print(link.attrs)       


# In[202]:


# list link
links = []
for link in html.findAll("a", href=get_href):
    links.append(link['href'])

print(links)


# In[199]:


# for link in links:
#     link = "https://edition.cnn.com" + link
#     html = requests.get(link).content
#     html = soup(html, 'html.parser')


#     for p in html.findAll("p"):
#         print(p.text.strip())
#     break
    


# In[200]:


import os
html = requests.get(cnn_url).content
html = soup(html, 'html.parser')

def get_href(link):
    return link and re.compile("index.html$").search(link)

links = []
for link in html.findAll("a", href=get_href):
    links.append("https://edition.cnn.com" + link['href'])

if not os.path.exists('BBC_weather'):
    os.mkdir('BBC_weather')

for i, link in enumerate(links):
    html = requests.get(link).content
    html = soup(html, 'html.parser')

    article_content = ""
    for p in html.findAll("p"):
        article_content += p.text.strip() + "\n"

    if article_content:
        with open(f'BBC_weather/{i+1}.txt', 'w', encoding='utf-8') as file:
            file.write(article_content)
        print(f'Lưu nội dung từ bài báo {i+1} vào Reuters_breakingviews/link_{i+1}.txt')

# text


# In[203]:


# Lấy tên link vào file .txt
links = []
for link in html.findAll("a", href=get_href):
    links.append("https://edition.cnn.com" + link['href'])

# Lưu các liên kết vào tệp txt
with open('links.txt', 'w') as f:
    for link in links:
        f.write("%s\n" % link)
    print(f'Lưu nội dung từ bài báo links.txt')


# In[ ]:




