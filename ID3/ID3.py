import math
import pandas as pd


class Node:
    def __init__(self, data, parent):
        self.data = data
        self.parent = parent
        self.children = []
        self.edge = None


class ID3:
    def __init__(self, df, label):
        self.attributes = dict()
        self.df = df
        self.label = label
        selected_cols = df[df.columns.difference([label])]

        for col in selected_cols.columns.values:
            self.attributes[col] = self.IG(col, label, df)
            # print('IG: {0}, {1}'.format(col, igs[col]))

    def fit(self, maxdepth=None):
        self.root = self.ID3(self.df, self.attributes, label, maxdepth)
        return self.root

    def ID3(self, S, attributes, label, maxdepth=None):
        if maxdepth is None:
            maxdepth = float('infinity')

        return self.ID3Rec(None, S, attributes, label, maxdepth)

    def ID3Rec(self, parent, S, attributes, label, depth):

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

        root = Node(A, parent)

        # for each possible value v of that A can take:
        for v in S[A].drop_duplicates().values:
            # node = Node(v, root)

            # Add a new tree branch corresponding to A=v
            # root.children.append(node)
            # node.value = v

            # Let Sv be the subset of examples in S with A=v
            Sv = self.getSubset(S, A, v, label)

            # if Sv is empty:
            if self.isSvEmpty(Sv, label):
                # add leaf node with the most common value of Label in S
                # node.children.append(Node(self.getMaxVal(Sv, label), root))
                root.children.append(Node(self.getMaxVal(Sv, label), root))
            else:
                # below this branch add the subtree ID3(Sv, Attributes - {A}, Label)
                # node.children.append(self.ID3Rec(node, Sv, self.diff(attributes, A), label, depth=depth - 1))
                node = self.ID3Rec(root, Sv, self.diff(attributes, A), label, depth=depth - 1)
                node.edge = v
                root.children.append(node)

        return root

    def getBestAttribute(self, attributes):
        return max(attributes, key=attributes.get)

    # def H(self, pos_proportion, neg_proportion):
    def H(self, proportions):
        # lg_pos = 0 if pos_proportion <= 0 else math.log(pos_proportion, 2)
        # lg_neg = 0 if neg_proportion <= 0 else math.log(neg_proportion, 2)

        sm = 0
        for prop in proportions:
            sm += -1 * prop * (0 if prop <= 0 else math.log(prop, 2))
        # return (-1 * pos_proportion * lg_pos) - (neg_proportion * lg_neg)
        return sm

    def proportions(self, y):
        props = []
        sm = 0
        for title, value in y.iteritems():
            props.append(value)
            sm += value

        return [x / sm for x in props]

    def IG(self, col, label, df):

        feature_set_x1 = df[col].drop_duplicates().values

        A = dict()
        for f in feature_set_x1:
            y = df[df[col] == f][label].value_counts()  # df[df[col] == f].values_counts()
            # pos, neg = (self.proportions(x))
            props = (self.proportions(y))
            # A[f] = self.H(pos, neg)
            A[f] = self.H(props)

        # full_0 = len(df[df[label] == 0][label]) / float(len(df))
        # full_1 = len(df[df[label] == 1][label]) / float(len(df))
        full_prop = self.proportions(df[label].value_counts())
        H_fullset = self.H(full_prop)

        sm = 0
        for v in A.keys():
            s_v = float(len(df[df[col] == v]))
            sm += (s_v / float(len(df))) * A[v]

        return H_fullset - sm

    def getSubset(self, S, A, v, label):
        return S[S[A] == v]

    def diff(self, attributes, A):
        attrs = attributes.copy()
        attrs.pop(A, None)
        attributes.pop(A, None)
        return attrs

    def isSvEmpty(self, Sv, label):
        return len(Sv[label].drop_duplicates()) == 0

    def getMaxVal(self, Sv, label):
        return Sv[label].value_counts().idxmax()

        # def predict(self, df):
        #     self.traverse(self.root, df)
        #
        # def traverse(self, node, df):
        #
        #     f = node.data
        #
        #     branch = df[node.data][0]
        #     for child in node.children:
        #
        #         if branch == child.data:
        #             self.traverse(child, df)


labels = set()


def generateGraph(root):
    print('digraph G {')
    count = 0
    if len(root.children) > 0:
        print(genGraphRec(node=root, count=count))
        for l in labels:
            print(l)
    else:
        print('"{0}"\n'.format(root.data))

    print('}')


def genGraphRec(node, count):
    string = ''

    if len(node.children) == 0:
        edge = '' if node.edge is None else node.edge
        labels.add('{1}{0}[label="{0}"];\n'.format(node.data, edge))

    for c in node.children:
        count += 1
        parent = '' if node.parent is None else node.parent.data
        edge = '' if node.edge is None else node.edge

        labels.add('{1}{0}[label="{0}"];\n'.format(node.data, edge))

        string += '"{1}{0}" -> "{3}{2}"[ label = "{3}"]"\n'. \
            format(node.data, edge, c.data, c.edge)

        string += genGraphRec(c, count)

    return string


def playTennis():
    d = dict(
        Outlook=['SUNNY', 'SUNNY', 'OVERCAST', 'RAIN', 'RAIN', 'RAIN', 'OVERCAST', 'SUNNY', 'SUNNY', 'RAIN', 'SUNNY',
                 'OVERCAST', 'OVERCAST', 'RAIN'],
        Temperature=['HOT', 'HOT', 'HOT', 'MILD', 'COOL', 'COOL', 'COOL', 'MILD', 'COOL', 'MILD', 'MILD', 'MILD', 'HOT',
                     'MILD'],
        Humidity=['HIGH', 'HIGH', 'HIGH', 'HIGH', 'NORMAL', 'NORMAL', 'NORMAL', 'HIGH', 'NORMAL', 'NORMAL', 'NORMAL',
                  'HIGH', 'NORMAL', 'HIGH'],
        Wind=['WEAK', 'STRONG', 'WEAK', 'WEAK', 'WEAK', 'STRONG', 'STRONG', 'WEAK', 'WEAK', 'WEAK', 'STRONG', 'STRONG',
              'WEAK', 'STRONG'],
        Play=['-', '-', '+', '+', '+', '-', '+', '-', '+', '+', '+', '+', '+', '-']
        # Play=[0,0,1,1,1,0,1,0,1,1,1,1,1,0]
    )

    predictand = 'Play'
    df = pd.DataFrame(data=d)

    df.loc[df[predictand] == '-', predictand] = 0
    df.loc[df[predictand] == '+', predictand] = 1

    return predictand, df, None


def shape():
    d = dict(
        color=['Blue', 'Green', 'Green', 'Green', 'Red', 'Red', 'Red', 'Blue', 'Blue', 'Blue', 'Blue'],
        shape=['Square', 'Circle', 'Square', 'Square', 'Square', 'Circle', 'Circle', 'Tri', 'Tri', 'Circle', 'Circle'],
        label=['A', 'A', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'C', 'C']
        # Play=[0,0,1,1,1,0,1,0,1,1,1,1,1,0]
    )

    test = dict(
        color=['Red'],
        shape=['Rect']
    )

    df = pd.DataFrame(data=d)
    test_df = pd.DataFrame(data=test)

    return 'label', df, test_df


def y():
    d = dict(x1=[0, 0, 0, 1, 0, 1, 0],
             x2=[0, 1, 0, 0, 1, 1, 1],
             x3=[1, 0, 1, 0, 1, 0, 0],
             x4=[0, 0, 1, 1, 0, 0, 1],
             y=[0, 0, 1, 1, 0, 0, 0])
    df = pd.DataFrame(data=d)
    label = 'y'
    return label, df, None


if __name__ == '__main__':
    # d = dict(
    #     O=['S', 'S', 'O', 'R', 'R', 'R', 'O', 'S', 'S', 'R', 'S', 'O', 'O', 'R'],
    #     T=['H', 'H', 'H', 'M', 'C', 'C', 'C', 'M', 'C', 'M', 'M', 'M', 'H', 'M'],
    #     H=['H', 'H', 'H', 'H', 'N', 'N', 'N', 'H', 'N', 'N', 'N', 'H', 'N', 'H'],
    #     W=['W', 'S', 'W', 'W', 'W', 'S', 'S', 'W', 'W', 'W', 'S', 'S', 'W', 'S'],
    #     Play=['-', '-', '+', '+', '+', '-', '+', '-', '+', '+', '+', '+', '+', '-']
    #     # Play=[0,0,1,1,1,0,1,0,1,1,1,1,1,0]
    # )

    # label, df = y()
    # label, df = triangle()
    # label, df = playTennis()
    # label, df = y()
    # label, df, test_df = shape()
    label, df, test_df = playTennis()

    id3 = ID3(df, label)

    root = id3.fit()

    # id3.predict(test_df)
    generateGraph(root)
