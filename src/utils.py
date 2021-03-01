import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.spatial import ConvexHull
from scipy.interpolate import Rbf

import src.constants as c


def point_in_hull(point, hull, tolerance=1e-12):
    """returns True iif the point is in a given convex hull

    Parameters
    ----------
    point : np.array of shape (D,)
        considered point
    hull : scipy.spatial.ConvexHull
        already computed convex hull
    tolerance : float, optional
        tolerance for computational imprecision, by default 1e-12

    Returns
    -------
    boolean
        whether or not the considered point lays within the convex hull
    """
    return all(
        (np.dot(eq[:-1], point) + eq[-1] <= tolerance)
        for eq in hull.equations)



def zoom_on_region(grid_inter, col_x, col_y, xlim, ylim, n_points):
    """zoom on a particular region defined by user with xlim and ylim widgets

    Parameters
    ----------
    grid_inter : pandas.DataFrame
        already filtered and averaged data grid
    col_x : string
        horizontal column in final plot
    col_y : string
        vertical column in final plot
    xlim : tuple (of two floats between 0 and 1)
        define the region to be plot (on horizontal axis).
        (0, 1) is the full region, and (.25, .75) would be the inner half 
    ylim : tuple (of two floats between 0 and 1)
        define the region to be plot (on vertical axis).
        (0, 1) is the full region, and (.25, .75) would be the inner half 
    n_points : int
        number of points to consider along one axis for the finer grid

    Returns
    -------
    grid_inter : pandas.DataFrame
        filtered and averaged and zoomed data grid 
    x : np.array of shape (n_points,)
        values at which the interpolation is to be evaluated for horizontal axis
    y : np.array of shape (n_points,)
        values at which the interpolation is to be evaluated for vertical axis
    """
    # compute horizontal limits of plot from grid and xlim input 
    x_min, x_max = grid_inter[col_x].min(), grid_inter[col_x].max()
    delta_x = x_max - x_min
    x_min_considered = x_min + xlim[0] * delta_x
    x_max_considered = x_max - (1 - xlim[1]) * delta_x
    x = np.linspace(x_min_considered, x_max_considered, n_points)
    
    # compute vertical limits of plot from grid and xlim input 
    y_min, y_max = grid_inter[col_y].min(), grid_inter[col_y].max()
    delta_y = y_max - y_min
    y_min_considered = y_min + ylim[0] * delta_y
    y_max_considered = y_max - (1 - ylim[1]) * delta_y
    y = np.linspace(y_min_considered, y_max_considered, n_points)
    
    # filter data grid
    grid_inter = grid_inter[
        (x_min_considered <= grid_inter[col_x])
        & (grid_inter[col_x] <= x_max_considered)
        & (y_min_considered <= grid_inter[col_y])
        & (grid_inter[col_y] <= y_max_considered)
    ]
    return grid_inter, x, y


def choose_grid_section(grid, col_x, col_y, col_z, value_coupe):
    """there are 3 input parameters in the full grid, and in this repo one can only
    look at 2 at the same time. This function sets where to look for 
    the 3rd one (low or high value)

    Parameters
    ----------
    grid : pandas.DataFrame
        full data grid
    col_x : string
        horizontal column in final plot
    col_y : string
        vertical column in final plot
    col_z : string
        column of which contours are to be plotted 
    value_coupe : string
        there are 3 input parameters in the full grid, and in this repo one can only
        look at 2 at the same time. This value sets where to
        look for the 3rd one (low or high value)

    Returns
    -------
    grid_inter : pandas.DataFrame
        filtered and averaged data grid 
    """
    if set([col_x, col_y]) == set(["[003]AVmax", "[001]Pressure"]):
        grid_inter = grid[grid["[002]radm"] == c.values_coupe["G0"][value_coupe]]
        grid_inter = grid_inter.groupby([col_x, col_y])[col_z].mean().reset_index()
    elif set([col_x, col_y]) == set(["[003]AVmax", "[002]radm"]):
        grid_inter = grid[grid["[001]Pressure"] == c.values_coupe["P"][value_coupe]]
        grid_inter = grid_inter.groupby([col_x, col_y])[col_z].mean().reset_index()
    elif set([col_x, col_y]) == set(["[002]radm", "[001]Pressure"]):
        grid_inter = grid[grid["[003]AVmax"] == c.values_coupe["Av"][value_coupe]]
        grid_inter = grid_inter.groupby([col_x, col_y])[col_z].mean().reset_index()
 
    # if the x axis and y axis are not input parameters, then there is not need to 
    # set any particlar value 
    else:
        grid_inter = grid.groupby([col_x, col_y])[col_z].mean().reset_index()

    return grid_inter


def compute_fig(grid, col_x, col_y, col_z, value_coupe, n_points, xlim, ylim):
    """uses user input parameters to compute the values to plot

    Parameters
    ----------
    grid : pandas.DataFrame
        full data grid
    col_x : string
        horizontal column in final plot
    col_y : string
        vertical column in final plot
    col_z : string
        column of which contours are to be plotted 
    value_coupe : string
        there are 3 input parameters in the full grid, and in this repo one can only
        look at 2 at the same time. This value sets where to
        look for the 3rd one (low or high value)
    n_points : int
        number of points to consider along one axis for the finer grid
    xlim : tuple (of two floats between 0 and 1)
        define the region to be plot (on horizontal axis).
        (0, 1) is the full region, and (.25, .75) would be the inner half 
    ylim : tuple (of two floats between 0 and 1)
        define the region to be plot (on vertical axis).
        (0, 1) is the full region, and (.25, .75) would be the inner half 

    Returns
    -------
    X : np.array of shape (n_points, n_points)
        x value corresponding to the interpolation evaluation in Z
    Y : np.array of shape (n_points, n_points)
        y value corresponding to the interpolation evaluation in Z
    Z : np.array of shape (n_points, n_points)
        interpolated values on the X Y finer grid 
    grid_inter : pandas.DataFrame
        filtered and averaged and zoomed data grid 
    """
    # 
    grid_inter = choose_grid_section(grid, col_x, col_y, col_z, value_coupe)
    
    # filter grid values to keep only those that are in the region specified by user
    grid_inter, x, y = zoom_on_region(grid_inter, col_x, col_y, xlim, ylim, n_points)

    # fit grid interpolator
    I = Rbf(*[grid_inter.iloc[:,col] for col in range(2)], grid_inter.iloc[:,-1], function="cubic")
    
    # evaluate interpolator on finer grid
    X, Y = np.meshgrid(x, y)
    XY = np.hstack([X.reshape((-1,1)), Y.reshape((-1,1))])
    Z = I(XY[:,0], XY[:,1])
    
    # filter points that are not in the convex hull of the grid
    # ie points for which the model is extrapolating and not interpolating
    # (since we use rbf interpolation, it is mathematecally possible)
    ch = ConvexHull(grid_inter.iloc[:,:2].values)
    for i,p in enumerate(XY):
        if not(point_in_hull(p, ch)):
            Z[i] = np.nan
    Z = Z.reshape((n_points,n_points))
        
    return X, Y, Z, grid_inter


def plot_fig(X, Y, Z, col_z, col_x, col_y, n_levels, grid_inter, show_data, filename, save_fig):
    """plots the results

    Parameters
    ----------
    X : np.array of shape (n_points, n_points)
        x value corresponding to the interpolation evaluation in Z
    Y : np.array of shape (n_points, n_points)
        y value corresponding to the interpolation evaluation in Z
    Z : np.array of shape (n_points, n_points)
        interpolated values on the X Y finer grid 
    col_z : string
        column for which the contours are drawn
    col_x : string
        name of horizontal axis
    col_y : string
        name of vertical axis
    n_levels : int
        number of levels to compute and draw
    grid_inter : pandas.DataFrame
        data grid 
    show_data : boolean
        whether or not to show the points of the data grid
    filename : string
        name of the file in which the figure is to be saved
    save_fig : boolean
        whether of not the figure is to be saved
    """
    fig, ax = plt.subplots(figsize=(8,8))
    CS = ax.contour(X, Y, Z, levels=n_levels)

    if show_data:
        ax.plot(grid_inter[col_x], grid_inter[col_y], "k+", label='grid')
        
    ax.clabel(CS, inline=1, fontsize=10)
    ax.set_title(col_z)
    ax.set_xlabel(col_x)
    ax.set_ylabel(col_y)
    
    plt.legend()
    plt.grid()
    
    if save_fig:
        if not(os.path.isdir(col_z)):
            os.mkdir(f"./{col_z}")
        plt.savefig(f"./{col_z}/{filename}")
    
    plt.show()