import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np

'''
    This package is to be used as a library. Please do not edit.
'''


def fpoly(x: float) -> float:
    """ Simple polynomial of degree 5"""
    return 0.009 * (x ** 5) + 0.02 * (x ** 4) - 0.32 * (x ** 3) - 0.54 * (x ** 2) + 3.2 * x - 1.0


def dfpoly(x: float) -> float:
    """Derivative of simple polynomial of degree 5"""
    return 5.0 * 0.009 * (x ** 4) + 4.0 * 0.02 * (x ** 3) - 3.0 * 0.32 * (x ** 2) - 2.0 * 0.54 * x + 3.2


# ======================================================================
# Functions for Newton Fractal

def generate_sampling(borders: list, size: int) -> np.ndarray:
    size_x = size
    size_y = int(size * (borders[3] - borders[2]) / (borders[1] - borders[0]))
    sx = np.linspace(borders[0], borders[1], size_x)
    sy = np.linspace(borders[2], borders[3], size_y)
    x, y = np.meshgrid(sx, sy)
    sampling = x + 1j * y
    return sampling, size_x, size_y


def get_colors(roots: np.ndarray) -> np.ndarray:
    colors = np.zeros((roots.shape[0], 3))
    c_idx = np.linspace(0.0, 1.0, roots.shape[0])
    cm = matplotlib.cm.get_cmap('jet')
    for idx, i in enumerate(c_idx):
        colors[idx] = cm(i)[:3]
    return colors



# Roots of unity
def rou(k):
    def f(x):
        return x ** k - 1

    return f


def drou(k):
    def f(x):
        return k * x ** (k - 1)

    return f


def rou_roots(k):
    return np.array([np.exp(2.j * np.pi * i / k) for i in range(k)])


rou_borders = [-1.5, 1.5, -1.5, 1.5]


# Polynomial
def poly(x):
    return x ** 3 - 2 * x + 2


def dpoly(x):
    return 3 * x ** 2 - 2


poly_roots = np.array([np.complex128(-1.76929235423863), np.complex128(0.884646177119316 + 0.589742805022206j),
                       np.complex128(0.884646177119316 - 0.589742805022206j)])
poly_borders = [-1.5, 0.5, -1.0, 1.0]


# Sinus function
def sin(x):
    return np.sin(x)


def dsin(x):
    return np.cos(x)


sin_roots = np.array(np.linspace(-10 * np.pi, 10 * np.pi, 21))
sin_borders = [-np.pi, np.pi, -np.pi, np.pi]

fractal_functions = [[rou(4), drou(4), rou_roots(4), rou_borders, "roots_of_unity_4"],
                     [rou(7), drou(7), rou_roots(7), rou_borders, "roots_of_unity_7"],
                     [poly, dpoly, poly_roots, poly_borders, "polynomial"],
                     [sin, dsin, sin_roots, sin_borders, "sinus"]]


# ======================================================================
# Functions for Minimal Surfaces

def generate_cylinder(nc, nz, scale=0.8):
    v = np.zeros((nc * nz, 3))
    f = np.zeros((2 * nc * (nz - 1), 3), dtype=int)

    phi = np.linspace(0.0, 2.0 * np.pi, endpoint=False, num=nc)
    z = np.linspace(0.0, 1.0, endpoint=True, num=nz)

    for i in range(nz):
        for j in range(nc):
            v[i * nc + j, :] = (scale * np.cos(phi[j]), scale * np.sin(phi[j]), z[i])

    for i in range(nz - 1):
        for j in range(nc):
            vi = i * nc + j
            ni = 1
            if j + 1 >= nc:
                ni -= nc
            f[2 * vi, :] = (vi, vi + ni, vi + nc)
            f[2 * vi + 1, :] = (vi + ni, vi + nc + ni, vi + nc)

    c1 = list(range(0, nc))
    c2 = list(range(v.shape[0] - nc, v.shape[0]))
    c1.extend(c2)
    c = np.array(c1)

    return v, f, c


def load_object(name):
    object = np.load(name + ".npz")
    return object["v"], object["f"], object["c"]


def prepare_visualization(v, f):
    fig = plt.figure()
    cmap = plt.get_cmap('Blues')
    norm = None
    ax = fig.gca(projection='3d')
    limits = (np.min(v[:, 0]), np.max(v[:, 0]), np.min(v[:, 1]), np.max(v[:, 1]), np.min(v[:, 2]), np.max(v[:, 2]))
    surf = ax.plot_trisurf(v[:, 0], v[:, 1], v[:, 2], triangles=f, cmap=cmap, linewidth=0.1, norm=norm, shade=False,
                           alpha=0.8)
    ax.set_xlim3d(limits[0], limits[1])
    ax.set_ylim3d(limits[2], limits[3])
    ax.set_zlim3d(limits[4], limits[5])
    ax.set_axis_off()
    return fig, surf, ax, limits

def update_visualization(v, f, abs_gradient, limits, ax, normalize=False):
    ax.clear()
    if normalize:
        abs_gradient = abs_gradient / abs_gradient.max()
    cmap = plt.get_cmap('Blues')
    norm = None
    surf = ax.plot_trisurf(v[:, 0], v[:, 1], v[:, 2], triangles=f, cmap=cmap, linewidth=0.1, norm=norm,
                           shade=False, alpha=0.8)
    surf.set_array(abs_gradient)
    ax.set_xlim3d(limits[0], limits[1])
    ax.set_ylim3d(limits[2], limits[3])
    ax.set_zlim3d(limits[4], limits[5])
    ax.set_axis_off()


def calculate_abs_gradient(g, f, c):
    indices = np.arange(g.shape[0])
    indices = np.delete(indices, c)
    grad_abs = np.sqrt(np.sum(g * g, axis=1))
    grad_per_tri = np.zeros(f.shape[0])
    for fa in range(f.shape[0]):
        fgrad = 0.0
        for idx in range(3):
            vidx = f[fa, idx]
            if vidx in indices:
                fgrad += grad_abs[vidx]
        grad_per_tri[fa] = fgrad

    return grad_per_tri
