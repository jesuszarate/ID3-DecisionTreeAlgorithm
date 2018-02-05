import math
import pandas as pd

class ID3:
    def __init__(self):
        pass

    def entropy(self, pos_proportion, neg_proportion):
        return (-1 * pos_proportion * math.log(pos_proportion, 2)) - \
               (neg_proportion * math.log(neg_proportion,2))

    def proportions(self, y):
        p = 0
        n = 0
        for i in range(len(y)):
            if y[i] == 0:
                n += 1
            else:
                p += 1
        sm = float(len(y))
        return p/sm, n/sm


if __name__ == '__main__':
    id3 = ID3()

    d = dict(x1=[0, 0, 0, 1, 0, 1, 0],
             x2=[0, 1, 0, 0, 1, 1, 1],
             x3=[1, 0, 1, 0, 1, 0, 0],
             x4=[0, 0, 1, 1, 0, 0, 1],
             y=[0, 0, 1, 1, 0, 0, 0])

    df = pd.DataFrame(data=d)


    x1 = df[df['x1'] == 0]['y'].values

    # print (x1)
    print(id3.proportions(x1))





