from typing import List

from comer.datamodules.hme100k.vocab import replace_textcircled_label


def main():

    with open(f'test_labels.txt', encoding='utf-8') as f:
        captions = f.readlines()

        for idx, line in enumerate(captions):
            tmp: List[str] = line.strip().split()
            label: List[str] = tmp[1:]
            file_name: str = tmp[0]
            stipped_file_name: str = tmp[0][:-4]

            label = replace_textcircled_label(label)

            with open(f"./test/{stipped_file_name}.tex", "wb") as f:
                f.write(generate_text(stipped_file_name, ' '.join(label)).encode(encoding="utf-8"))

def generate_text(fname: str, label: str):
    text = "%" + fname + """
\\documentclass[11pt,a4paper]{scrreprt}
\\usepackage{extarrows}
\\begin{document}\n$""" + label + "$\n\end{document}"
    return text

if __name__ == '__main__':
    main()