############################################
# imports
############################################
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patheffects
from matplotlib import collections as mcoll
import glob
import numpy as np
import Protein

pymol_color_list = ["#33ff33","#00ffff","#ff33cc","#ffff00","#ff9999","#e5e5e5","#7f7fff","#ff7f00",
                    "#7fff7f","#199999","#ff007f","#ffdd5e","#8c3f99","#b2b2b2","#007fff","#c4b200",
                    "#8cb266","#00bfbf","#b27f7f","#fcd1a5","#ff7f7f","#ffbfdd","#7fffff","#ffff7f",
                    "#00ff7f","#337fcc","#d8337f","#bfff3f","#ff7fff","#d8d8ff","#3fffbf","#b78c4c",
                    "#339933","#66b2b2","#ba8c84","#84bf00","#b24c66","#7f7f7f","#3f3fa5","#a5512b"]

pymol_cmap = matplotlib.colors.ListedColormap(pymol_color_list)

##################################################
# plotting
##################################################
def kabsch(a, b, weights=None, return_v=False):
  a = np.asarray(a)
  b = np.asarray(b)
  if weights is None: weights = np.ones(len(b))
  else: weights = np.asarray(weights)
  B = np.einsum('ji,jk->ik', weights[:, None] * a, b)
  u, s, vh = np.linalg.svd(B)
  if np.linalg.det(u @ vh) < 0: u[:, -1] = -u[:, -1]
  if return_v: return u
  else: return u @ vh

def plot_pseudo_3D(xyz, c=None, ax=None, chainbreak=5,
                   cmap="gist_rainbow", line_w=2.0,
                   cmin=None, cmax=None, zmin=None, zmax=None):

  def rescale(a,amin=None,amax=None):
    a = np.copy(a)
    if amin is None: amin = a.min()
    if amax is None: amax = a.max()
    a[a < amin] = amin
    a[a > amax] = amax
    return (a - amin)/(amax - amin)

  # make segments
  xyz = np.asarray(xyz)
  seg = np.concatenate([xyz[:-1,None,:],xyz[1:,None,:]],axis=-2)
  seg_xy = seg[...,:2]
  seg_z = seg[...,2].mean(-1)
  ord = seg_z.argsort()

  # set colors
  if c is None: c = np.arange(len(seg))[::-1]
  else: c = (c[1:] + c[:-1])/2
  c = rescale(c,cmin,cmax)  

  if isinstance(cmap, str):
    if cmap == "gist_rainbow": c *= 0.75
    colors = matplotlib.cm.get_cmap(cmap)(c)
  else:
    colors = cmap(c)
  
  if chainbreak is not None:
    dist = np.linalg.norm(xyz[:-1] - xyz[1:], axis=-1)
    colors[...,3] = (dist < chainbreak).astype(np.float)

  # add shade/tint based on z-dimension
  z = rescale(seg_z,zmin,zmax)[:,None]
  tint, shade = z/3, (z+2)/3
  colors[:,:3] = colors[:,:3] + (1 - colors[:,:3]) * tint
  colors[:,:3] = colors[:,:3] * shade

  set_lim = False
  if ax is None:
    fig, ax = plt.subplots()
    fig.set_figwidth(5)
    fig.set_figheight(5)
    set_lim = True
  else:
    fig = ax.get_figure()
    if ax.get_xlim() == (0,1):
      set_lim = True
      
  if set_lim:
    xy_min = xyz[:,:2].min() - line_w
    xy_max = xyz[:,:2].max() + line_w
    ax.set_xlim(xy_min,xy_max)
    ax.set_ylim(xy_min,xy_max)

  ax.set_aspect('equal')
    
  # determine linewidths
  width = fig.bbox_inches.width * ax.get_position().width
  linewidths = line_w * 72 * width / np.diff(ax.get_xlim())

  lines = mcoll.LineCollection(seg_xy[ord], colors=colors[ord], linewidths=linewidths,
                               path_effects=[matplotlib.patheffects.Stroke(capstyle="round")])
  
  return ax.add_collection(lines)



def add_text(text, ax):
  return plt.text(0.5, 1.01, text, horizontalalignment='center',
                  verticalalignment='bottom', transform=ax.transAxes)

def plot_protein(protein=None, pos=None, plddt=None, Ls=None, dpi=100, best_view=True, line_w=2.0):
  
  if protein is not None:
    pos = np.asarray(protein.atom_positions[:,1,:])
    plddt = np.asarray(protein.b_factors[:,0])

  # get best view
  if best_view:
    if plddt is not None:
      weights = plddt/100
      pos = pos - (pos * weights[:,None]).sum(0,keepdims=True) / weights.sum()
      pos = pos @ kabsch(pos, pos, weights, return_v=True)
    else:
      pos = pos - pos.mean(0,keepdims=True)
      pos = pos @ kabsch(pos, pos, return_v=True)

  if plddt is not None:
    fig, (ax1, ax2) = plt.subplots(1,2)
    fig.set_figwidth(6); fig.set_figheight(3)
    ax = [ax1, ax2]
  else:
    fig, ax1 = plt.subplots(1,1)
    fig.set_figwidth(3); fig.set_figheight(3)
    ax = [ax1]
    
  fig.set_dpi(dpi)
  fig.subplots_adjust(top = 0.9, bottom = 0.1, right = 1, left = 0, hspace = 0, wspace = 0)

  xy_min = pos[...,:2].min() - line_w
  xy_max = pos[...,:2].max() + line_w
  for a in ax:
    a.set_xlim(xy_min, xy_max)
    a.set_ylim(xy_min, xy_max)
    a.axis("off")

  if Ls is None or len(Ls) == 1:
    # color N->C
    c = np.arange(len(pos))[::-1]
    plot_pseudo_3D(pos,  line_w=line_w, ax=ax1)
    add_text("colored by N→C", ax1)
  else:
    # color by chain
    c = np.concatenate([[n]*L for n,L in enumerate(Ls)])
    if len(Ls) > 40:   plot_pseudo_3D(pos, c=c, line_w=line_w, ax=ax1)
    else:              plot_pseudo_3D(pos, c=c, cmap=pymol_cmap, cmin=0, cmax=39, line_w=line_w, ax=ax1)
    add_text("colored by chain", ax1)
    
  if plddt is not None:
    # color by pLDDT
    plot_pseudo_3D(pos, c=plddt, cmin=50, cmax=90, line_w=line_w, ax=ax2)
    add_text("colored by pLDDT", ax2)
  return fig




if "__main__"==__name__:
  pdbfiles = glob.glob('*.pdb')
  for pdbfile in pdbfiles:
    _protein = Protein.from_pdb_filename(pdbfile)
    fig = plot_protein(_protein, dpi=320,line_w=1.5)
    fig.savefig("{}.png".format(pdbfile))
    print("{}.png".format(pdbfile))
    plt.close()