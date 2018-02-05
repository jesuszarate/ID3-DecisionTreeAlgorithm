import math
import pandas as pd


class ID3:
    def __init__(self):
        pass

    def H(self, pos_proportion, neg_proportion):

        lg_pos = 0 if pos_proportion <= 0 else math.log(pos_proportion, 2)
        lg_neg = 0 if neg_proportion <= 0 else math.log(neg_proportion, 2)

        return (-1 * pos_proportion * lg_pos) - \
               (neg_proportion * lg_neg)

    def proportions(self, y):
        p = 0
        n = 0
        for i in range(len(y)):
            if y[i] == 0:
                n += 1
            else:
                p += 1
        sm = float(len(y))
        return p / sm, n / sm

    def IG(self, col, predictand, df):

        feature_set_x1 = df[col].drop_duplicates().values

        arr = dict()
        for f in feature_set_x1:
            x = df[df[col] == f][predictand].values
            pos, neg = (id3.proportions(x))
            arr[f] = id3.H(pos, neg)

        full_0 = len(df[df[predictand] == 0][predictand]) / float(len(df))
        full_1 = len(df[df[predictand] == 1][predictand]) / float(len(df))

        # y_vals = df['y'].drop_duplicates().values
        # for i in y_vals:
        #     full = len(df[df['y'] == i]['y'])/float(len(df))

        H_fullset = id3.H(full_1, full_0)

        # print ('full: {0}'.format(H_fullset))

        sm = 0
        for i in range(len(arr)):
            s_v = float(len(df[df[col] == i]))  # Todo: this will only work for x1 right now as is
            sm += (s_v / float(len(df))) * arr[i]

        # print ('Sum: {0}'.format(sm))
        # print ('IG: {0}'.format(H_fullset - sm))
        return H_fullset - sm


if __name__ == '__main__':
    id3 = ID3()

    d = dict(x1=[0, 0, 0, 1, 0, 1, 0],
             x2=[0, 1, 0, 0, 1, 1, 1],
             x3=[1, 0, 1, 0, 1, 0, 0],
             x4=[0, 0, 1, 1, 0, 0, 1],
             y=[0, 0, 1, 1, 0, 0, 0])

    df = pd.DataFrame(data=d)

    for col in df.columns.values[:-1]:
        print ('IG: {0}, {1}'.format(col, id3.IG(col, 'y', df)))

