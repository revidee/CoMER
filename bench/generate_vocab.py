from collections import defaultdict


def main():

    train_labels = open('train_labels.txt', 'r', encoding="utf8")
    test_labels = open('test_labels.txt', 'r', encoding="utf8")

    vocab_set = defaultdict(int)
    kanji_start = "丙"
    total_kanji_labels = 0
    tot_kanjis = 0

    for label_file in [train_labels, test_labels]:
        line_no = 0
        for line in label_file:
            line_no += 1
            line = line.strip()
            label_start = line.find('\t')
            if label_start == -1:
                print(f'no tab found at line {line_no}')
                exit(0)
            label = line[label_start + 1:]
            tokens = label.split(' ')
            had_kanji = False
            for token in tokens:
                vocab_set[token] += 1
                if token >= kanji_start:
                    if not had_kanji:
                        total_kanji_labels += 1
                        had_kanji = True
                    tot_kanjis += 1

    vocab_list = list(vocab_set.keys())
    for sorted_key in sorted(vocab_list):
        print(sorted_key, vocab_set[sorted_key])
    print('total kanji', tot_kanjis)
    print('total kanji labels', total_kanji_labels)



if __name__ == '__main__':
    main()