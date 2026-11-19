import phoebe
from phoebe import u
from phoebe.atmospheres import models
from astropy.io import fits
import os


path = os.path.dirname(__file__)
atms = [atm for atm in models.get_supported_atms(include_extern=False)]


def test_compute(atm_tables_available=False, atm_path=None, atms=atms):
    pb = phoebe.atmospheres.passbands.Passband(
        ptf=os.path.join(path, 'test.ptf'),
        pbset='test',
        pbname='t1',
        wlunits=u.nm,
        calibrated=True,
        reference='test passband creation, loading and saving',
        version=1.0
    )

    # generate blackbody tables:
    pb.compute_intensities(atm=models.BlackbodyModelAtmosphere(), include_mus=False, include_ld=False, verbose=True)

    if atm_tables_available:
        # generate tables:
        for atm in atms:
            if atm.name == 'blackbody':
                continue

            instance = atm.from_path(atm_path + '/' + atm.name)
            pb.compute_intensities(atm=instance, include_extinction=True, verbose=True)

    pb.save(os.path.join(path, 'test.fits'))


def test_load():
    pb = phoebe.atmospheres.passbands.Passband.load(os.path.join(path, 'test.fits'))
    assert 'blackbody:Inorm' in pb.content

    # cleanup
    os.remove(os.path.join(path, 'test.fits'))


def test_skip_obsolete(tmp_path, capsys):
    pb = phoebe.atmospheres.passbands.Passband(
        ptf=os.path.join(path, 'test.ptf'),
        pbset='test',
        pbname='obsolete',
        wlunits=u.nm,
        calibrated=True,
        reference='test skipping obsolete passbands on initialization',
        version=1.0
    )
    pb.compute_intensities(atm=models.BlackbodyModelAtmosphere(), include_mus=False, include_ld=False)

    # _init_passband identifies obsolete files by their PHOEBEVN header; files
    # created before 2.6 (pre-2.5 format or the old phoenix grid) are skipped:
    archive = os.path.join(tmp_path, 'test_obsolete.fits')
    pb.save(archive)
    fits.setval(archive, 'PHOEBEVN', value='2.5.5')
    phoebe.atmospheres.passbands._init_passband(archive)
    assert 'test:obsolete' not in phoebe.atmospheres.passbands._pbtable
    assert 'not available from the server' in capsys.readouterr().out

    # 2.6 development versions are current:
    fits.setval(archive, 'PHOEBEVN', value='2.6.0.dev0')
    phoebe.atmospheres.passbands._init_passband(archive)
    assert 'test:obsolete' in phoebe.atmospheres.passbands._pbtable
    phoebe.atmospheres.passbands._pbtable.pop('test:obsolete')

if __name__ == '__main__':
    atm_path = os.path.join(path, 'tables')
    atm_tables_available = os.path.exists(atm_path)
    test_compute(atm_tables_available=atm_tables_available, atm_path=atm_path)
    test_load()
    test_skip_obsolete()
