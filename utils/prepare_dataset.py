import os
import shutil

# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from config import CFG


def split_by_label(ls_img):
    dict_label = dict(zip(CFG["labels"], [[] for _ in range(len(CFG["labels"]))]))
    for label in CFG["labels"]:
        for img in ls_img:
            if label in img.lower():
                dict_label[label] = dict_label.get(label, []) + [img]
    for label in CFG["labels"]:
        print(f"{label}: {len(dict_label[label])}")
    return dict_label


def mkdir():
    for dir in ["train", "val", "test"]:
        for label in CFG["labels"]:
            os.makedirs(f'{CFG["data_dir"]}/{dir}/{label}', exist_ok=True)


def move_img(dict_label):
    for label in CFG["labels"]:
        for img in dict_label[label]:
            if img in dict_label[label][: int(len(dict_label[label]) * 0.8)]:
                shutil.copy(
                    f'{CFG["data_dir"]}/img/{img}',
                    f'{CFG["data_dir"]}/train/{label}/{img}',
                )
            elif (
                img
                in dict_label[label][
                    int(len(dict_label[label]) * 0.8) : int(
                        len(dict_label[label]) * 0.9
                    )
                ]
            ):
                shutil.copy(
                    f'{CFG["data_dir"]}/img/{img}',
                    f'{CFG["data_dir"]}/val/{label}/{img}',
                )
            else:
                shutil.copy(
                    f'{CFG["data_dir"]}/img/{img}',
                    f'{CFG["data_dir"]}/test/{label}/{img}',
                )


if __name__ == "__main__":
    ls_img = os.listdir(f'{CFG["data_dir"]}/img')
    dict_label = split_by_label(ls_img)
    mkdir()
    move_img(dict_label)
