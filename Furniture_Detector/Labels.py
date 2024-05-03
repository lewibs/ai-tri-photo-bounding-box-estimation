labels = ["__background__", "chair", "couch", "table", "bookshelf", "desk", "bed", "dresser", "ottoman", "tv stand", "other"]
label_indexs = [i for i in range(len(labels))]

def label_to_index(label):
    return labels.index(label)

def index_to_label(index):
    return labels[index]

