import pandas as pd

# transform original .txt files from the dataset into expected .csv format for the code

## annotations

path = "/datasets/home/hbenoit/DivDis-exp/data/list_attr_celeba.txt"

attr=[]

with open(path) as f:
    out = f.read().split("\n")
    num_images = int(out[0])
    columns = ["image_id"] + out[1].split(" ")[:-1]



    for i in range(2, num_images+2):
        img_attr = list(filter(lambda x: x != "", out[i].strip().split(" ")))
        for i in range(1, len(img_attr)):
            img_attr[i] = int(img_attr[i])
        attr.append(img_attr)


df = pd.DataFrame(attr, columns=columns)

df.to_csv(path.replace(".txt",".csv"))


### partition files

path = "/datasets/home/hbenoit/DivDis-exp/data/list_eval_partition.txt"

df = pd.read_fwf(path, header=None, )
df.columns = columns=["image_id","partition"]

df.to_csv(path.replace(".txt",".csv"), index=False)