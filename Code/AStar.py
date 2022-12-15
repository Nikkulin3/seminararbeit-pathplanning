import pickle
from collections import deque

from Code.Configurations import ConfigurationSpace


class Graph:
    def __init__(self):
        self.c = ConfigurationSpace()
        try:
            with open("graph.pkl", "rb") as f:
                self.adjac_lis: dict = pickle.load(f)
        except FileNotFoundError:
            allowed = {}
            for i, j in self.c.free_tree.query_pairs(self.c.resolution * 1.5):
                if i not in allowed:
                    allowed[i] = []
                allowed[i].append(j)
            self.adjac_lis: dict = allowed
            with open("graph.pkl", "wb") as f:
                pickle.dump(allowed, f)
    def get_neighbors(self, v):
        try:
            neigh = self.adjac_lis[v]
        except KeyError:
            neigh = []
        return ((n, self.probability_of(n)) for n in neigh)

        # This is heuristic function which is having equal values for all nodes

    def probability_of(self, n):
        return 1
        # H = {
        #     'A': 1,
        #     'B': 1,
        #     'C': 1,
        #     'D': 1
        # }
        #
        # return H[n]

    def a_star_algorithm(self, start, stop):
        # In this open_lst is a lisy of nodes which have been visited, but who's
        # neighbours haven't all been always inspected, It starts off with the start
        # node
        # And closed_lst is a list of nodes which have been visited
        # and who's neighbors have been always inspected
        open_lst = set([start])
        closed_lst = set([])

        # poo has present distances from start to all other nodes
        # the default value is +infinity
        poo = {}
        poo[start] = 0

        # par contains an adjac mapping of all nodes
        par = {}
        par[start] = start

        while len(open_lst) > 0:
            n = None

            # it will find a node with the lowest value of f() -
            for v in open_lst:
                if n is None or poo[v] + self.probability_of(v) < poo[n] + self.probability_of(n):
                    n = v

            if n is None:
                print('Path does not exist!')
                return None

            # if the current node is the stop
            # then we start again from start
            if n == stop:
                reconst_path = []

                while par[n] != n:
                    reconst_path.append(n)
                    n = par[n]

                reconst_path.append(start)

                reconst_path.reverse()

                print('Path found: {}'.format(reconst_path))
                return reconst_path

            # for all the neighbors of the current node do
            print(self.c.free_space_l[n])
            for (m, weight) in self.get_neighbors(n):
                # if the current node is not present in both open_lst and closed_lst
                # add it to open_lst and note n as it's par
                if m not in open_lst and m not in closed_lst:
                    open_lst.add(m)
                    par[m] = n
                    poo[m] = poo[n] + weight

                # otherwise, check if it's quicker to first visit n, then m
                # and if it is, update par data and poo data
                # and if the node was in the closed_lst, move it to open_lst
                else:
                    if poo[m] > poo[n] + weight:
                        poo[m] = poo[n] + weight
                        par[m] = n

                        if m in closed_lst:
                            closed_lst.remove(m)
                            open_lst.add(m)

            # remove n from the open_lst, and add it to closed_lst
            # because all of his neighbors were inspected
            open_lst.remove(n)
            closed_lst.add(n)

        print('Path does not exist!')
        return None


if __name__ == '__main__':
    Graph().a_star_algorithm(0, 1228541)
