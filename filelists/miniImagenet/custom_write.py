import os
import json
import pandas as pd

train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("val.csv")
test_df = pd.read_csv("test.csv")

label_names = []
for set_df in [train_df, val_df, test_df]:
    label_names += set_df["label"].unique().tolist()

base = {"label_names": label_names}
novel = {"label_names": label_names}
val = {"label_names": label_names}

abs_path = os.path.abspath(".")
class_label = 0
for col, set_df in zip([base, val, novel], [train_df, val_df, test_df]):
    image_names = []
    image_labels = []
    for i, row in set_df.iterrows():
        image_names.append(os.path.join(abs_path, "images", row["filename"]))
        image_labels.append(class_label // 600)
        class_label += 1
    col["image_names"] = image_names
    col["image_labels"] = image_labels

with open("base.json", "w") as f:
    json.dump(base, f)

with open("novel.json", "w") as f:
    json.dump(novel, f)

with open("val.json", "w") as f:
    json.dump(val, f)

