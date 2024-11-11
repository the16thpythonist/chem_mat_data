import re
import io
import types
import typing as t
from typing import Union

import cairosvg
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from rdkit import Chem
from rdkit.Chem.Draw.rdMolDraw2D import MolDraw2DSVG
from rdkit.Chem import rdDepictor
from chem_mat_data.graph import assert_graph_dict

rdDepictor.SetPreferCoordGen(True)


def create_frameless_figure(width: int = 100,
                            height: int = 100,
                            ratio: int = 2,
                            dim: int = 2,
                            show_spines: bool = False,
                            show_axis: bool = False,
                            ) -> t.Tuple[plt.Figure, plt.Axes]:
    """
    Returns a tuple of a matplotlib Figure and Axes object, where the axes object is a complete blank slate
    that can act as the foundation of a matplotlib-based visualization of a graph.

    More specifically this means that upon saving the figure that is created by this function, there will
    be no splines for the axes, not any kind of labels, no background, nothing at all. When saving this
    figure it will be a completely transparent picture with the pixel size given by ``width`` and ``height``.

    :param int width: The width of the saved image in pixels
    :param int height: The height of the saved image in pixels
    :param float ratio: This ratio will change the internal matplotlib figure size but *not* the final size
        of the image. This will be important for example if there is text with a fixed font size within the
        axes. This value will affect the size of things like text, border widths etc. but not the actual
        size of the image.
    :return:
    """
    
    fig = plt.figure(figsize=(width / (100 * ratio), height / (100 * ratio)))
    fig.set_dpi(100 * ratio)
    
    if dim == 2:
        ax = fig.add_subplot(111)
    elif dim == 3:
        ax = fig.add_subplot(111, projection='3d')

    # https://stackoverflow.com/questions/14908576
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    if dim == 3:
        ax.get_zaxis().set_ticks([])
        
        # Also: In the 3D case we always want to show the axis, because 3D structures are not really 
        # interpretable without them.
        show_axis = True

    ax.spines['top'].set_visible(show_spines)
    ax.spines['right'].set_visible(show_spines)
    ax.spines['bottom'].set_visible(show_spines)
    ax.spines['left'].set_visible(show_spines)
    if not show_axis:
        ax.axis('off')

    # Selecting the axis-X making the bottom and top axes False.
    plt.tick_params(axis='x', which='both', bottom=False,
                    top=False, labelbottom=False)

    # Selecting the axis-Y making the right and left axes False
    plt.tick_params(axis='y', which='both', right=False,
                    left=False, labelleft=False)

    # https://stackoverflow.com/questions/4581504
    fig.patch.set_facecolor((0, 0, 0, 0))
    fig.patch.set_visible(False)

    ax.patch.set_facecolor((0, 0, 0, 0))
    ax.patch.set_visible(False)

    # A big part of achieving the effect we desire here, that is having only the Axes show up in the final
    # file and none of the border or padding of the Figure, is which arguments are passed to the "savefig"
    # method of the figure object. Since the saving process will come later we make sure that the correct
    # parameters are used by overriding the default parameters for the savefig method here
    def savefig(this, *args, **kwargs):
        this._savefig(*args, dpi=100 * ratio, **kwargs)

    setattr(fig, '_savefig', fig.savefig)
    setattr(fig, 'savefig', types.MethodType(savefig, fig))
    
    fig.tight_layout(pad=0, h_pad=0, w_pad=0)

    return fig, ax


def visualize_molecular_graph_from_mol(ax: plt.Axes,
                                       mol: Chem.Mol,
                                       image_width: 1000,
                                       image_height: 1000,
                                       line_width: int = 5,
                                       ) -> t.Tuple[np.ndarray, str]:
    """
    Creates a molecular graph visualization if given the RDKit Mol object ``mol`` and the matplotlib Axes
    ``ax`` to draw on. The image width and height have to be the same values as the final pixel values of
    the rendered PNG matplotlib figure.

    Returns a tuple, where the first value is the ``node_positions`` array of shape (V, 2) where V is the
    number of nodes in the graph (number of atoms in the molecule). This array is created alongside the
    visualization and for every atom it contains the (x, y) coordinates on the given Axes.

    NOTE: The node positions returned by this function are in the coordinates system of the given Axes
        object. When intending to save that into a persistent file it is important to convert these
        node coordinates into the Figure coordinate system first by using ax.transData.transform !

    05.06.23 - Previously, this function relied on the usage of a temp dir and created two temporary files
        as intermediates. This was now replaces such that no intermediate files are required anymore to
        improve the efficiency of the function.

    :param ax: The mpl Axes object onto which the visualization should be drawn
    :param mol: The Mol object which is to be visualized
    :param image_width: The pixel width of the resulting image
    :param image_height: The pixel height of the resulting image
    :param line_width: Defines the line width used for the drawing of the bonds

    :return: A tuple (node_positions, svg_string), where the first element is a numpy array (V, 2) of node
        mpl coordinates of each of the graphs nodes in the visualization on the given Axes and the second
        element is the SVG string from which that visualization was created.
    """
    # To create the visualization of the molecule we are going to use the existing functionality of RDKit
    # which simply takes the Mol object and creates an SVG rendering of it.
    mol_drawer = MolDraw2DSVG(image_width, image_height)
    mol_drawer.SetLineWidth(line_width)
    mol_drawer.DrawMolecule(mol)
    mol_drawer.FinishDrawing()
    svg_string = mol_drawer.GetDrawingText()

    # Now the only problem we have with the SVG that has been created this way is that it still has a white
    # background, which we generally don't want for the graph visualizations and sadly there is no method
    # with which to control this directly for the drawer class. So we have to manually edit the svg string
    # to get rid of it...
    svg_string = re.sub(
        r'opacity:\d*\.\d*;fill:#FFFFFF',
        'opacity:0.0;fill:#FFFFFF',
        svg_string
    )

    # Now, we can't directly display SVG to a matplotlib canvas, which is why we first need to convert this
    # svg string into a PNG image file temporarily which we can then actually put onto the canvas.
    png_data = cairosvg.svg2png(
        bytestring=svg_string.encode(),
        parent_width=image_width,
        parent_height=image_height,
        output_width=image_width,
        output_height=image_height,
    )
    file_obj = io.BytesIO(png_data)

    image = Image.open(file_obj, formats=['png'])
    image = np.array(image)
    ax.imshow(image)

    # The RDKit svg drawer class offers some nice functionality to figure out the coordinates of those
    # files within the drawer.
    node_coordinates = []
    for point in [mol_drawer.GetDrawCoords(i) for i, _ in enumerate(mol.GetAtoms())]:
        node_coordinates.append([
            point.x,
            point.y
        ])

    node_coordinates = np.array(node_coordinates)
    
    # 23.10.23 - Here we just want to make sure that we are properly doing the memory management of 
    # for this function.
    file_obj.close()
    del image, mol_drawer, png_data

    return node_coordinates, svg_string


# This is simply a wrapper function of the function above to make it more convenient to use since here we 
# want to support different types of input for the molecule - including directly passing graph dicts
def plot_molecule(ax: plt.Axes,
                  molecule: Union[Chem.Mol, str, dict],
                  image_width: int = 1000,
                  image_height: int = 1000,
                  **kwargs,
                  ) -> np.ndarray:
    """
    Plot a visual representation of the given ``molecule`` on the given matplotlib Axes object ``ax``.
    
    :param ax: The matplotlib Axes object onto which the visualization should be drawn
    :param molecule: The molecule that should be visualized. This can be either a RDKit Mol object, a SMILES
        string or a graph dict that contains a SMILES string under the key "graph_repr".
    :param image_width: The pixel width of the resulting image
    :param image_height: The pixel height of the resulting image
    
    :return: A numpy array of shape (V, 2) containing the node positions of the molecule in the visualization.
        This can be used to plot additional information on top of the molecule visualization.
    """
    
    # ~ data validation
    # Since we allow multiple different data types as the input for the molecule here we will perform some 
    # validation to make sure that the input actually specifies a valid molecule that can be plotted at all.
    
    if isinstance(molecule, str):
        mol = Chem.MolFromSmiles(molecule)
        assert mol, f'Could not convert SMILES string "{molecule}" into a valid RDKit molecule object!'
        
    elif isinstance(molecule, dict):
        assert_graph_dict(molecule)
        assert 'graph_repr' in molecule, 'The given graph dict does not contain a "graph_repr" (smiles) key!'
        mol = Chem.MolFromSmiles(molecule['graph_repr'])
        assert mol, (
            f'Could not convert the SMILES string "{molecule["graph_repr"]}" from the given graph dict '
            f'into a valid RDKit molecule object!'
        )
    
    elif isinstance(molecule, Chem.Mol):
        mol = molecule
        
    else:
        raise TypeError(f'Unsupported molecule type "{type(molecule)}" for plotting!')
    
    # ~ visualization
    # for the visualization itself we can just use this function which handles the actual plotting based on the 
    # rdkit Mol object.
    
    node_coordinates, _ = visualize_molecular_graph_from_mol(
        ax=ax,
        mol=mol,
        image_width=image_width,
        image_height=image_height,
        **kwargs,
    )
    
    return node_coordinates