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
# bem = mne.make_bem_solution(surfs)

# code for ECoG forward follows
import numpy as np
from mne.bem import _lin_pot_coeff, _check_complete_surface

els = (surfs[0]['rr'] + surfs[1]['rr']) / 2.
els = els[:5]  # random points between 2 surfs for testing

nps = [surf['np'] for surf in surfs]
np_tot = sum(nps)
coeffs = np.zeros((len(els), np_tot))
offsets = np.cumsum(np.concatenate(([0], nps)))

sigma = np.r_[0.0, conductivity]

v_tot = np.zeros((len(els),))
for idx, surf in enumerate(surfs):
    _check_complete_surface(surf)  # needed to get tri_area of decimated surf
    o1, o2 = offsets[idx], offsets[idx + 1]
    for k in range(surf['ntri']):
        tri_rr = surf['rr'][surf['tris']][k]
        tri_area = surf['tri_area'][k]
        tri_nn = surf['tri_nn'][k]
        tri = surf['tris'][k]
        coeff = _lin_pot_coeff(els, tri_rr, tri_nn, tri_area)
        coeffs[:, offsets[o1:o2]][:, tri] -= coeff

    # see _bem_specify_coils
    v_surf = np.dot(coeff, bem['solution'][o1:o2])
    v_surf *= (sigma[idx + 1] - sigma[idx])  # check
    v_tot += v_surf
# sigmas = # get sigmas according to els
v_tot *= -sigmas / (4 * np.pi)

# _get_inf_pots

# use _CheckInside()(els) for getting sigma

sdfdfdf
fwd = mne.make_forward_solution(raw_fname, trans=trans, src=src, bem=bem,
                                meg=True, eeg=False, mindist=5.0, n_jobs=1,
                                verbose=True)
