import requests
from bs4 import BeautifulSoup
from googlesearch import search
import re
import pandas as pd
import os
from tqdm import tqdm as tqdm


def search_google(query, num=10):
    wiki = "wikipedia"
    final_query = query + " " + wiki
    search_results = search(query=final_query, num=num, start=0, stop=num, lang="en", pause=2.0)
    return search_results


def scrape_wikipedia_page(url):
    response = requests.get(url)
    html = str(response.text)
    soup = BeautifulSoup(html, 'html.parser')
    try:
        title = soup.find("h1").get_text()
    except AttributeError:
        title = ""
    text_lines = soup.find_all("p")
    text = []
    for line in text_lines:
        line_text = line.get_text()
        line_text = re.sub("\[[0-9]*]", "", line_text)
        text.append(line_text)
    return title, text


def search_and_scrape_topic(topic):
    texts, titles, topics = [], [], []
    search_results = search_google(query=topic)
    for result_url in search_results:
        if "wikipedia" not in result_url or "#" in result_url:
            continue
        title, text = scrape_wikipedia_page(result_url)
        texts.append(text)
        titles.append(title)
        topics.append(topic)
    return texts, titles, topics


def search_and_scrape_from_csv(path, path_to_save="datasets/scraped_data"):
    scrape_df = pd.DataFrame()
    topics_list = pd.read_csv(path, header=None)
    topics_list = list(topics_list[0])
    text, title, topic = [], [], []
    for topic_ in tqdm(topics_list, total=len(topics_list)):
        texts, titles, topics = search_and_scrape_topic(topic_)
        text.extend(texts)
        title.extend(titles)
        topic.extend(topics)

    scrape_df["text"] = text
    scrape_df["title"] = title
    scrape_df["topic"] = topic
    print(scrape_df)
    print(len(scrape_df))

    pickle_path_to_save = path_to_save + ".pkl"
    csv_path_to_save = path_to_save + ".csv"
    scrape_df.to_pickle(pickle_path_to_save)
    scrape_df.to_csv(csv_path_to_save)


if __name__ == '__main__':
    path = "datasets/all_topics.csv"
    search_and_scrape_from_csv(path)
