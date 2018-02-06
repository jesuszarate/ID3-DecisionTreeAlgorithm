import math
import pandas as pd


class Node:
    def __init__(self, data, parent):
        self.data = data
        self.parent = parent
        self.children = []


class ID3:

    def ID3(self, S, attributes, label, maxdepth = None):
        if maxdepth == None:
            maxdepth = float('infinity')

        return self.ID3Rec(S, attributes, label, maxdepth)

    def ID3Rec(self, S, attributes, label, depth):


        # if all examples have same label:
        if len(S[label].drop_duplicates()) == 1 or depth == 0:
            # return a leaf node with the label ;
            return Node(S[label].drop_duplicates().values[0], None)

        # if Attributes empty:
        if not attributes:
            # return a leaf node with the most common label
            return Node(self.getMaxVal(S, label), None)

        # Create a Root node for tree
        A = self.getBestAttribute(attributes)  # attribute in Attributes that best splits S

        root = Node(A, None)

        # for each possible value v of that A can take:
        for v in S[A].drop_duplicates().values:
            node = Node(v, root)
            root.children.append(node)
            # Add a new tree branch corresponding to A=v

            # Let Sv be the subset of examples in S with A=v
            Sv = self.getSubset(S, A, v, label)

            # if Sv is empty:
            if self.isSvEmpty(Sv, label):
                # add leaf node with the most common value of Label in S
                node.children.append(Node(self.getMaxVal(Sv, label), root))
            else:
                # below this branch add the subtree ID3(Sv, Attributes - {A}, Label)
                node.children.append(self.ID3Rec(Sv, self.diff(attributes, A), label, depth=depth-1))

        return root

    def getBestAttribute(self, attributes):
        return max(attributes, key=attributes.get)

    def H(self, pos_proportion, neg_proportion):
        lg_pos = 0 if pos_proportion <= 0 else math.log(pos_proportion, 2)
        lg_neg = 0 if neg_proportion <= 0 else math.log(neg_proportion, 2)

        return (-1 * pos_proportion * lg_pos) - (neg_proportion * lg_neg)

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

        A = dict()
        for f in feature_set_x1:
            x = df[df[col] == f][predictand].values
            pos, neg = (self.proportions(x))
            A[f] = self.H(pos, neg)

        full_0 = len(df[df[predictand] == 0][predictand]) / float(len(df))
        full_1 = len(df[df[predictand] == 1][predictand]) / float(len(df))

        H_fullset = self.H(full_1, full_0)

        sm = 0
        for v in A.keys():
            s_v = float(len(df[df[col] == v]))
            sm += (s_v / float(len(df))) * A[v]

        return H_fullset - sm

    def spliton(self, df, predictand):
        igs = dict()
        selected_cols = df[df.columns.difference([predictand])]

        for col in selected_cols.columns.values:
            igs[col] = self.IG(col, predictand, df)
            print('IG: {0}, {1}'.format(col, igs[col]))

        # return max(igs, key=igs.get)
        return igs

    def getSubset(self, S, A, v, label):
        # print(S)
        # print(S[S[A] == v])
        return S[S[A] == v]

    def diff(self, attributes, A):
        attributes.pop(A, None)
        return attributes

    def isSvEmpty(self, Sv, label):
        return len(Sv[label].drop_duplicates()) == 0

    # def getMostCommon(self, Sv, v):
    #     Sv[]

    def getMaxVal(self, Sv, label):
        return Sv.loc[Sv[label].idxmax()][label]


def generateGraph(root):

    print('digraph G {')

    if len(root.children) > 0:
        print(genGraphRec(node=root))
    else:
        print('"{0}"\n'.format(root.data))

    print('}')


def genGraphRec(node):

    string = ''
    for c in node.children:
        parent = '' if node.parent is None else node.parent.data
        string += '"{0}:{1}" -> "{1}:{2}"\n'.format(parent, node.data, c.data)
        string += genGraphRec(c)

    return string

if __name__ == '__main__':
    d = dict(x1=[0, 0, 0, 1, 0, 1, 0],
             x2=[0, 1, 0, 0, 1, 1, 1],
             x3=[1, 0, 1, 0, 1, 0, 0],
             x4=[0, 0, 1, 1, 0, 0, 1],
             y=[0, 0, 1, 1, 0, 0, 0])

    # d = dict(
    #     O=['S', 'S', 'O', 'R', 'R', 'R', 'O', 'S', 'S', 'R', 'S', 'O', 'O', 'R'],
    #     T=['H', 'H', 'H', 'M', 'C', 'C', 'C', 'M', 'C', 'M', 'M', 'M', 'H', 'M'],
    #     H=['H', 'H', 'H', 'H', 'N', 'N', 'N', 'H', 'N', 'N', 'N', 'H', 'N', 'H'],
    #     W=['W', 'S', 'W', 'W', 'W', 'S', 'S', 'W', 'W', 'W', 'S', 'S', 'W', 'S'],
    #     Play=['-', '-', '+', '+', '+', '-', '+', '-', '+', '+', '+', '+', '+', '-']
    #     # Play=[0,0,1,1,1,0,1,0,1,1,1,1,1,0]
    # )

    # d = dict(
    #     Overcast=['SUNNY', 'SUNNY', 'OVERCAST', 'RAIN', 'RAIN', 'RAIN', 'OVERCAST', 'SUNNY', 'SUNNY', 'RAIN', 'SUNNY', 'OVERCAST', 'OVERCAST', 'RAIN'],
    #     Temperature=['HOT', 'HOT', 'HOT', 'MILD', 'COOL', 'COOL', 'COOL', 'MILD', 'COOL', 'MILD', 'MILD', 'MILD', 'HOT', 'MILD'],
    #     Humidity=['HIGH', 'HIGH', 'HIGH', 'HIGH', 'NORMAL', 'NORMAL', 'NORMAL', 'HIGH', 'NORMAL', 'NORMAL', 'NORMAL', 'HIGH', 'NORMAL', 'HIGH'],
    #     Wind=['WEAK', 'STRONG', 'WEAK', 'WEAK', 'WEAK', 'STRONG', 'STRONG', 'WEAK', 'WEAK', 'WEAK', 'STRONG', 'STRONG', 'WEAK', 'STRONG'],
    #     Play=['-', '-', '+', '+', '+', '-', '+', '-', '+', '+', '+', '+', '+', '-']
    #     # Play=[0,0,1,1,1,0,1,0,1,1,1,1,1,0]
    # )

    predictand = 'y'
    df = pd.DataFrame(data=d)

    # df.loc[df[predictand] == '-', predictand] = 0
    # df.loc[df[predictand] == '+', predictand] = 1

    id3 = ID3()
    atts = id3.spliton(df, predictand)

    root = id3.ID3(df, atts, predictand)

    generateGraph(root)