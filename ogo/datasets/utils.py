import dgl
import itertools


def ogo_batcher():
    def collate_fn(batch: list):
        # each batch contains a list of graphs, so we merge the lists together to form a super list of graphs
        graphs = itertools.chain(batch)
        graphs = dgl.batch(graphs)
        return graphs
    return collate_fn
