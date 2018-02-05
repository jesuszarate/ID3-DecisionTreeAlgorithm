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

    def IG(self, df):

        feature_set_x1 = df['x1'].drop_duplicates().values

        arr = []
        for f in feature_set_x1:
            x = df[df['x1'] == f]['y'].values
            pos, neg = (id3.proportions(x))
            arr.append(id3.H(pos, neg))

        full_0 = len(df[df['y'] == 0]['y'])/float(len(df))
        full_1 = len(df[df['y'] == 1]['y'])/float(len(df))

        # y_vals = df['y'].drop_duplicates().values
        # for i in y_vals:
        #     full = len(df[df['y'] == i]['y'])/float(len(df))

        H_fullset = id3.H(full_1, full_0)

        print ('full: {0}'.format(H_fullset))

        sm = 0
        for i in range(len(arr)):
            s_v = float(len(df[df['x1'] == i])) # Todo: this will only work for x1 right now as is
            sm += (s_v/float(len(df))) * arr[i]

        print ('Sum: {0}'.format(sm))

        print ('IG: {0}'.format(H_fullset - sm))



if __name__ == '__main__':
    id3 = ID3()

    d = dict(x1=[0, 0, 0, 1, 0, 1, 0],
             x2=[0, 1, 0, 0, 1, 1, 1],
             x3=[1, 0, 1, 0, 1, 0, 0],
             x4=[0, 0, 1, 1, 0, 0, 1],
             y=[0, 0, 1, 1, 0, 0, 0])

    df = pd.DataFrame(data=d)



    # x1_0 = df[df['x1'] == 0]['y'].values
    # pos0, neg0 = (id3.proportions(x1_0))
    # print(id3.H(pos0,neg0))
    #
    # x1_1 = df[df['x1'] == 1]['y'].values
    # pos1, neg1 = (id3.proportions(x1_1))
    # print(id3.H(pos1,neg1))
    # print
    # arr = [id3.H(pos0, neg0), id3.H(pos1, neg1)]

    id3.IG(df)

