import csv


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


if __name__ == '__main__':
    select_only_over_thresh("datasets/clean_dataset.csv", "datasets/over_1000.csv", 1000)
