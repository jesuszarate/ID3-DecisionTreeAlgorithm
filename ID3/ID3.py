import math
import pandas as pd
from graphviz import Digraph


class Node:
    def __init__(self, data, parent):
        self.data = data
        self.parent = parent
        self.children = []
        self.edge = None

    def __str__(self):
        return str(self.data)


latex_newline = '\\\\'


class ID3:
    def __init__(self, df, label):
        self.attributes = dict()
        self.df = df
        self.label = label
        self.logging = False
        selected_cols = df[df.columns.difference([label])]

        self.root = None

        for col in selected_cols.columns.values:
            self.attributes[col] = self.IG(col, label, df)

    def fit(self, maxdepth=None, logging=False):
        self.logging = logging
        self.root = self.ID3(self.df, self.attributes, label, maxdepth)
        return self.root

    def ID3(self, S, attributes, label, maxdepth=None):
        if maxdepth is None:
            maxdepth = float('infinity')


        return self.ID3Rec(None, S, S, attributes, label, maxdepth)

    def ID3Rec(self, parent, full_S, S, attributes, label, depth):

        # if all examples have same label:
        if len(S[label].drop_duplicates()) == 1 or depth == 0:
            # return a leaf node with the label ;
            node_data = S[label].drop_duplicates().values[0]
            if self.logging:
                edge = S[parent.data].drop_duplicates().values[0]
                print('For edge {0} creating leaf node $\\rightarrow${1}{2}'.format(edge, node_data, latex_newline))
                print('-' * 60 + latex_newline)
            return Node(node_data, parent)

        # if Attributes empty:
        if not attributes:
            node_data = self.getMaxVal(S, label)
            if self.logging:
                edge = S[parent.data].drop_duplicates().values[0]
                print('For edge {0} creating leaf node $\\rightarrow${1}{2}'.format(edge, node_data, latex_newline))
                print('-' * 60 + latex_newline)

            # return a leaf node with the most common label
            return Node(node_data, parent)

        # Create a Root node for tree
        A = self.getBestAttribute(S, label)  # attribute in Attributes that best splits S
        if self.logging:
            print('Split on $\\rightarrow$ {0}{1}'.format(A, latex_newline))

        root = Node(A, parent)

        # for each possible value v of that A can take:
        for v in full_S[A].drop_duplicates().values:
        # for v in S[A].drop_duplicates().values:
            # node = Node(v, root)

            # Add a new tree branch corresponding to A=v

            # Let Sv be the subset of examples in S with A=v
            Sv = self.getSubset(S, A, v, label)

            # if Sv is empty:
            if self.isSvEmpty(Sv, label):
                node = Node(self.getMaxVal(S, label), root)
                node.edge = v
                # add leaf node with the most common value of Label in S
                root.children.append(node)

                # d = S[S[A].value_counts().idxmax()]
                # root.children.append(Node(d, root))
            else:
                # below this branch add the subtree ID3(Sv, Attributes - {A}, Label)
                node = self.ID3Rec(root, full_S, Sv, self.diff(attributes, A), label, depth=depth - 1)
                node.edge = v
                root.children.append(node)  # Add a new tree branch corresponding to A=v

        return root

    def getBestAttribute(self, S, label):
        selected_cols = S[S.columns.difference([label])]
        attributes = dict()

        for col in selected_cols.columns.values:
            attributes[col] = self.IG(col, label, S)

        if self.logging:
            print('MAX IG: {0}{1}'.format(max(attributes, key=attributes.get), latex_newline))
            print('*' * 50 + latex_newline)
        return max(attributes, key=attributes.get)

    def majority_error(self, S, attr, label):
        majority = S[S[label] == attr]
        return 1 - majority

    def H(self, proportions):
        '''
        Entropy
        :param proportions:
        :return the entropy:
        '''
        sm = 0
        for prop in proportions:
            sm += -1 * prop * (0 if prop <= 0 else math.log(prop, 2))
        return sm

    def proportions(self, y):
        props = []
        sm = 0
        for title, value in y.iteritems():
            props.append(value)
            sm += value

        return [x / sm for x in props]

    def frac_proportions(self, y, frac):
        props = []
        sm = 0
        for title, value in y.iteritems():
            props.append(value)
            sm += value

        return [(x+frac) / (sm+frac) for x in props]


    def IG(self, col, label, df):

        feature_set_x1 = df[col].drop_duplicates().values

        A = dict()
        for f in feature_set_x1:
            y = df[df[col] == f][label].value_counts()

            if 'None' in feature_set_x1:
                frac = len(df[df[col]==f])/(len(df[label]) - 1)
                props = (self.frac_proportions(y, frac))
            else:
                props = (self.proportions(y))

            A[f] = self.H(props)

        full_prop = self.proportions(df[label].value_counts())
        H_fullset = self.H(full_prop)

        exp_entropy = self.expected_entropy(A, col)

        information_gain = H_fullset - exp_entropy

        if self.logging:
            info = dict(entropy=A,
                        ExpectedEntropy=exp_entropy,
                        InformationGain=information_gain)
            self.logInfoGain(col, info)

        return information_gain

    def expected_entropy(self, A, col):
        sm = 0
        for v in A.keys():
            s_v = float(len(df[df[col] == v]))
            sm += (s_v / float(len(df))) * A[v]
        return sm

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

    def predict(self, df):
        return self.traverse(self.root, df)

    def traverse(self, node, df):

        if len(node.children) == 0:
            return node

        branch = df[node.data][0]
        for child in node.children:

            if branch == child.edge:
                result_node = self.traverse(child, df)
                return child if result_node is None else result_node
        return None

    def logInfoGain(self, col, info):
        print()
        print('Feature = {0}{1}'.format(col, latex_newline))

        for key, val in info.items():

            if key == 'entropy':
                for k, v in val.items():
                    print('$H_{{{0}}} = {1}${2}'.format(k, v, latex_newline))
            else:
                print('{0} $= {1}${2}'.format(key, val, latex_newline))


def generateGraph(root):
    dot = Digraph(comment='Graph')
    if len(root.children) > 0:
        genGraphRec(dot, node=root)
    else:
        dot.node(str(root.data))

    dot.render('graph.gv', view=True)


def genGraphRec(dot, node):
    # Leaf nodes
    if len(node.children) == 0:
        edge = '' if node.edge is None else node.edge
        par = '{1}{0}'.format(node.data, edge)
        dot.node(par, str(node.data))
        return

    for c in node.children:
        edge = '' if node.edge is None else node.edge

        par = '{1}{0}'.format(node.data, edge)
        kid = '{1}{0}'.format(c.data, c.edge)
        dot.node(par, str(node.data))

        # dot.render('graph.gv', view=True)
        # dot.node(kid, str(c.data))
        dot.edge(par, kid, str(c.edge))

        genGraphRec(dot, c)


def addAttributes(d, new_item):
    for k, val in new_item.items():
        d[k].append(val)

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

    new_item = dict(
        # Outlook='OVERCAST',
        Outlook='None',
        Temperature='MILD',
        Humidity='NORMAL',
        Wind='WEAK',
        Play='+'
    )

    # addAttributes(d, new_item)

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
        color=['Green'],
        shape=['Tri']
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
    # label, df = y()
    # label, df = triangle()
    # label, df = playTennis()
    # label, df, test_df = y()
    label, df, test_df = shape()
    # label, df, test_df = playTennis()

    id3 = ID3(df, label)
    root = id3.fit(logging=True)

    print('Should be in class: {0}'.format(id3.predict(test_df)))
    generateGraph(root)

    # fun()[p[c
