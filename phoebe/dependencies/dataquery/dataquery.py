import phoebe


class Query(object):
    pass


class GaiaQuery(Query):
    def __init__(self):
        from astroquery.gaia import Gaia
        self.query = Gaia
        self.query.MAIN_GAIA_TABLE = 'gaiadr3.gaia_source'
        self.tables = [table.get_qualified_name() for table in self.query.load_tables(only_names=True)]

    def login(self, username, password):
        pass

    def download_data(self, dr3ids):
        record = self.query.load_data(ids=dr3ids, data_release='Gaia DR3', retrieval_type='EPOCH_PHOTOMETRY', data_structure='COMBINED')
        for datatype in record:
            entry = record[datatype][0].array
            source = entry['source_id'].compressed()
            filter = entry['band'].compressed()
            time = entry['time'].compressed()
            flux = entry['flux'].compressed()
            ferr = entry['flux_error'].compressed()

            available_sources = set(source)
            available_filters = set(filter)

            data = {}
            for src in available_sources:
                src_mask = source == src
                for flt in available_filters:
                    flt_mask = filter == flt
                    lc = {}
                    lc['time'] = time[src_mask & flt_mask]
                    lc['flux'] = flux[src_mask & flt_mask]
                    lc['ferr'] = ferr[src_mask & flt_mask]

                    data[f'{src}:{flt}'] = lc
        return data

    def add_datasets_to_bundle(self, bundle, dr3ids):
        data = self.download_data(dr3ids=dr3ids)
        for lcid, lc in data.items():
            source_id, filter = lcid.split(':')
            bundle.add_dataset('lc', times=lc['time'], fluxes=lc['flux'], sigmas=lc['ferr'], passband=f'Gaia:{filter}')
