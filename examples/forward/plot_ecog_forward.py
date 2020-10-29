import os.path as op
import mne
from mne.datasets import sample

data_path = sample.data_path()

# the raw file containing the channel location + types
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
# The paths to Freesurfer reconstructions
subjects_dir = data_path + '/subjects'
subject = 'sample'

# The transformation file obtained by coregistration
trans = data_path + '/MEG/sample/sample_audvis_raw-trans.fif'

info = mne.io.read_info(raw_fname)

src = mne.setup_source_space(subject, spacing='oct4', add_dist='patch',
                             subjects_dir=subjects_dir)

surface = op.join(subjects_dir, subject, 'bem', 'inner_skull.surf')

# conductivity = (0.3,)  # for single layer
conductivity = (0.3, 0.006, 0.3)  # for three layers
surfs = mne.make_bem_model(subject='sample', ico=4,
                           conductivity=conductivity,
                           subjects_dir=subjects_dir)
bem = mne.make_bem_solution(surfs)

# code for ECoG/sEEG forward follows
import numpy as np
from mne.bem import _lin_pot_coeff, _check_complete_surface

els = (surfs[0]['rr'] + surfs[1]['rr']) / 2.
els = els[:5]  # random points between 2 surfs for testing

nps = [surf['np'] for surf in surfs]
np_tot = sum(nps)
coeffs = np.zeros((len(els), np_tot))
offsets = np.cumsum(np.concatenate(([0], nps)))

sigma = np.r_[0.0, conductivity]

for idx, surf in enumerate(surfs):
    _check_complete_surface(surf)  # needed to get tri_area of decimated surf
    o1, o2 = offsets[idx], offsets[idx + 1]
    for k in range(surf['ntri']):
        tri_rr = surf['rr'][surf['tris']][k]
        tri_area = surf['tri_area'][k]
        tri_nn = surf['tri_nn'][k]
        tri = surf['tris'][k]
        coeff = _lin_pot_coeff(els, tri_rr, tri_nn, tri_area)
        coeffs[:, o1:o2][:, tri] -= coeff

    coeffs[:, o1:o2] *= bem['field_mult'][idx]  # sigma+ - sigma-

# similar to _bem_specify_coils
sol = np.dot(coeffs, bem['solution'])

# source_mult = 2. / (sigma+ - sigma-)
# mults.shape = (1, n_surf_vertices)
mults = np.repeat(bem['source_mult'] / (-4.0 * np.pi),
                  [len(s['rr']) for s in bem['surfs']])[np.newaxis, :]
sol *= mults

# add 1/sigma(r) ??
from mne.surface import _CheckInside
check_insides = [_CheckInside(surf) for surf in surfs]
for el_idx, el in enumerate(els):
    # go from inside to outside.
    # if that's how conductivities are arranged?
    for sigma, check_inside in zip(conductivity[::-1], check_insides[::-1]):
        # check_inside accepts vector, so there might be
        # a better approach but let's go with dumb approach first
        if check_inside(el[None, :])[0]:
            sol[el_idx] += 1. / sigma
            break


# get the final gain matrix?
from mne.transforms import read_trans, apply_trans
from mne.forward._compute_forward import _bem_inf_pots

trans = read_trans(trans)
rr = np.concatenate([s['rr'][s['vertno']] for s in src])
mri_rr = np.ascontiguousarray(apply_trans(trans, rr))

bem_rr = np.concatenate([s['rr'] for s in bem['surfs']])
mri_Q = trans['trans'][:3, :3].T

v0s = _bem_inf_pots(mri_rr, bem_rr, mri_Q)
v0s = v0s.reshape(-1, v0s.shape[2])
G = np.dot(v0s, sol.T)
