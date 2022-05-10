"""
Module that enables individual plots to io data via matplotlib inheritance.

For more info, `see here <https://matplotlib.org/stable/gallery/subplots_axes_and_figures/custom_figure_class.html>`_.
"""

from matplotlib.figure import Figure


# Not sure if this is needed...
class IndividualDataPlot(Figure):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

