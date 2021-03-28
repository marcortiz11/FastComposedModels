import Source.pytorch as systorch


# Exchange chains between two ensembles
def single_point_crossover(sys1: systorch.System, sys2: systorch.System, chain1: systorch.Chain, chain2: systorch.Chain):

    def search_and_update_references_graph(ref, chain_replaced, chain_added):
        if isinstance(ref, systorch.Merger):
            for idx, component in enumerate(ref.get_merged_modules()):
                if component == chain_replaced:
                    ref.get_merged_modules()[idx] = chain_added
                else:
                    search_and_update_references_graph(component, chain_replaced, chain_added)
        elif isinstance(ref, systorch.Chain):
            for idx, component in enumerate(ref.get_chained_modules()):
                if component == chain_replaced:
                    ref.get_chained_modules()[idx] = chain_added
                else:
                    search_and_update_references_graph(component, chain_replaced, chain_added)
        elif isinstance(ref, systorch.System):
            if ref.get_start() == chain_replaced:
                ref.set_start(chain_added)
            else:
                search_and_update_references_graph(ref.get_start(), chain_replaced, chain_added)

    search_and_update_references_graph(sys1, chain1, chain2)
    search_and_update_references_graph(sys2, chain2, chain1)
