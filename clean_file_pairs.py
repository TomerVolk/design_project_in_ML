import csv
import pandas as pd


def clean_file(in_path, out_path):
    ans = []
    with open(in_path, newline='', encoding="UTF-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',', quotechar='"')
        for idx, row in enumerate(csv_reader):
            if idx == 0:
                continue
            arg1, arg2, label, acceptance, topic, _, arg1_stance, arg2_stance = row
            label = int(label)
            acceptance = float(acceptance)
            if acceptance < 0.8:
                continue
            if arg1_stance != arg2_stance:
                continue
            if label == 1:
                winner = arg1
                loser = arg2
                pass
            else:
                winner = arg2
                loser = arg1
            ans.append((winner, loser, topic))
    print("Read File")
    with open(out_path, "w", newline='', encoding="UTF-8") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(["Winner", "Loser", "Topic"])
        for row in ans:
            csv_writer.writerow(row)


def flatten_file(in_path, out_path):
    data = pd.read_csv(in_path, index_col=0)
    data1 = data.copy()
    data2 = data1.copy()
    data1 = data1.drop(columns='Loser').rename(columns={'Winner': 'Argument'})
    data2 = data2.drop(columns='Winner').rename(columns={'Loser': 'Argument'})
    data = data1.append(data2)
    data.drop_duplicates(subset=['Argument'], inplace=True)
    num_items = data[['Counter', 'Hyper_topic']].groupby('Hyper_topic').count()
    data.drop(columns=['Counter'], inplace=True)
    data = data.join(num_items, on='Hyper_topic')
    data.to_csv(out_path, index=False)


def clean_flatten_file(in_path: str):
    df = pd.read_csv(in_path)
    df = df.drop_duplicates(subset=['Argument'])
    out_path = in_path.replace('.csv', ' clean.csv')
    df.to_csv(out_path)


def select_only_over_thresh(in_path, out_path, thresh):
    ans = []
    hyper_topics = set()
    with open(in_path, newline='', encoding="UTF-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',', quotechar='"')
        for idx, row in enumerate(csv_reader):
            if idx == 0:
                continue
            _, winner, loser, topic, counter, hyper_topic = row
            counter = int(counter)
            if counter > thresh:
                hyper_topics.add(hyper_topic)
                ans.append((winner, loser, hyper_topic))
    print("Read File")
    with open(out_path, "w", newline='', encoding="UTF-8") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(["Winner", "Loser", "Hyper_Topic"])
        for row in ans:
            csv_writer.writerow(row)


def select_topics(topics, in_path, out_path):
    ans = []
    with open(in_path, newline='', encoding="UTF-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',', quotechar='"')
        for idx, row in enumerate(csv_reader):
            if idx == 0:
                continue
            winner, loser, topic = row
            if topic in topics:
                ans.append((winner, loser, topic))
    print("Read File")
    with open(out_path, "w", newline='', encoding="UTF-8") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(["Winner", "Loser", "Hyper_Topic"])
        for row in ans:
            csv_writer.writerow(row)


def read_arg_file(path, out_path):
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset=['argument'])
    df = df.dropna()
    num_items = df[['topic', 'argument']].groupby('topic').count().rename(columns={'argument': 'count'})
    df = df.join(num_items, on='topic')
    df.to_csv('datasets/arg quality with topic counts.csv')
    num_items.to_csv('datasets/arg quality topic counts.csv')
    # topics = set(df['topic'].tolist())
    # topics = sorted(list(topics))
    # topics_file = pd.DataFrame({'topic': topics})
    # topics_file.to_csv('datasets/topics names arg quality.csv', index=False)
    pass


if __name__ == '__main__':
    read_arg_file('datasets/arg_quality_rank_30k.csv', 'datasets/arg quality clean.csv')
    # flatten_file('datasets/clean_dataset.csv', 'datasets/pairs sentences.csv')
    # clean_flatten_file('datasets/pairs sentences.csv')
    # select_topics(["Vegetarianism"], "datasets/over_1000.csv", "datasets/single_topic.csv")
