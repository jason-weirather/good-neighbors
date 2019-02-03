import h5py, json
import pandas as pd
import numpy as np

class GoodNeighbors(object):
    def __init__(self,h5path,location='',mode='r'):
        self.h5path = h5path
        self.mode = mode
        self.location = location
        self._distance = None
        if mode in ['w','r+','a']:
            f = h5py.File(self.h5path,mode)
            if location != '': f.create_group(location)
            dset = f.create_dataset('/meta', (100,), dtype=h5py.special_dtype(vlen=str))
            dset.attrs['microns_per_pixel'] = np.nan
            dset.attrs['groupby'] = json.dumps([])
            dset.attrs['radius'] = np.nan
            #dset.attrs['id'] = uuid4().hex
            f.close()

    @property
    def neighborhood_fractions(self):
        return pd.read_hdf(self.h5path,self.location+'/cells/fractions')

    def set_neighborhood_fractions(self,radius):
        self.radius = radius
        dist = self.measure_distance()
        focus = dist.loc[dist['distance']<radius]
        totals = focus.groupby(self.groupby+['phenotype_label','cell_index']).count()[['db_id']].\
            rename(columns={'db_id':'total'}).reset_index()
        indiv = focus.groupby(self.groupby+['phenotype_label','cell_index','n_phenotype_label']).count()[['db_id']].\
            rename(columns={'db_id':'count'}).reset_index()
        fracs = totals.merge(indiv,on=self.groupby+['phenotype_label','cell_index'])
        fracs['fraction'] = fracs.apply(lambda x: x['count']/x['total'],1)
        fracs = fracs.set_index(self.groupby+['phenotype_label','cell_index','total'])\
            [['n_phenotype_label','fraction']].\
            pivot(columns='n_phenotype_label')
        fracs.columns = fracs.columns.droplevel(0)
        fracs = fracs.fillna(0).stack().reset_index().rename(columns={0:'fraction'})
        fracs.to_hdf(self.h5path,self.location+'/cells/fractions',
                     mode='r+',format='table',complib='zlib',complevel=9)
        fracs['radius'] = radius
        return 

    def plot_counts_per_radius(self,min_radius=0,max_radius=150,step_radius=2):
        dist = self.measure_distance()
        max_distance = 150
        #radius = 71 
        mpp = 0.496
        census = []
        for i in range(1,min(max_distance,int(dist['distance'].max())),2):
            one = dist.loc[dist['distance']<i].\
                groupby(['db_id']).count()[['n_db_id']].\
                rename(columns={'n_db_id':'count'}).reset_index()[['count']].\
                quantile([0.05,0.25,0.5,0.75,0.95],0)
            one.index.name = 'quantile'
            one = one.reset_index()
            one['distance'] = i
            one = one.fillna(0)
            census.append(one)
        census = pd.concat(census).pivot(columns='quantile',index='distance',values='count')
        census.columns = [str(x) for x in census.columns]
        census = census.reset_index()
        return census
        
    def measure_distance(self):
        if self._distance is not None: return self._distance
        one = self.cells.copy().reset_index(drop=True)
        one.index.name = 'db_id'
        one = one.reset_index()
        dist = one.merge(one.rename(columns={'x':'n_x','y':'n_y',
                              'phenotype_label':'n_phenotype_label',
                              'cell_index':'n_cell_index','db_id':'n_db_id'}),on=self.groupby)
        dist = dist.loc[dist['db_id']!=dist['n_db_id']]
        xdist = dist['x'].subtract(dist['n_x'])
        ydist = dist['y'].subtract(dist['n_y'])
        dist['distance'] = np.sqrt(xdist.multiply(xdist).add(ydist.multiply(ydist)))
        self._distance = dist
        return dist

    @property
    def radius(self):
        f = h5py.File(self.h5path,'r')
        levels = (self.location+'/meta').split('/')
        d = f
        for level in levels:
            if level == '': continue
            d = d[level]
        return d.attrs['radius']
    @radius.setter
    def radius(self,value):
        if not self.mode in ['a','r+','w']: raise ValueError('cant write on readonly')
        f = h5py.File(self.h5path,'r+')
        levels =(self.location+'/meta').split('/')
        d = f
        for level in levels:
            if level == '': continue
            d = d[level]
        d.attrs['radius'] = value
        return
        
    @property
    def microns_per_pixel(self):
        f = h5py.File(self.h5path,'r')
        levels = (self.location+'/meta').split('/')
        d = f
        for level in levels:
            if level == '': continue
            d = d[level]
        return d.attrs['microns_per_pixel']
    @microns_per_pixel.setter
    def microns_per_pixel(self,value):
        if not self.mode in ['a','r+','w']: raise ValueError('cant write on readonly')
        f = h5py.File(self.h5path,'r+')
        levels =(self.location+'/meta').split('/')
        d = f
        for level in levels:
            if level == '': continue
            d = d[level]
        d.attrs['microns_per_pixel'] = value
        return
        
    @property
    def groupby(self):
        f = h5py.File(self.h5path,'r')
        levels = (self.location+'/meta').split('/')
        d = f
        for level in levels:
            if level == '': continue
            d = d[level]
        return json.loads(d.attrs['groupby'])
    @groupby.setter
    def groupby(self,value):
        if not self.mode in ['a','r+','w']: raise ValueError('cant write on readonly')
        f = h5py.File(self.h5path,'r+')
        levels =(self.location+'/meta').split('/')
        d = f
        for level in levels:
            if level == '': continue
            d = d[level]
        d.attrs['groupby'] = json.dumps(value)
        return
    

    @property
    def cells(self):
        if self._cells is not None: return self._cells.copy()
        f = h5py.File(self.h5path,'r')
        names = [x for x in f]
        if self.location != '':
            d = f
            levels = self.location.split('/')
            for i, level in enumerate(levels):
                if level == '': continue
                d = d[level]
                names = [x for x in f]
        f.close()
        if 'cells' not in names: return None
        self._cells = pd.read_hdf(self.h5path,self.location+'/cells')
        return self._cells.copy()


    def set_cells(self,df,groupby):
        if self.mode not in ['w','r+','a']: raise ValueError("Cannot set data in read-only")
        self.groupby = groupby
        required = ['cell_index','x','y','phenotype_label']
        for r in required: 
            if r not in df.columns: raise ValuError("Cell dataframe input needs at least "+str(required))
        df = pd.DataFrame(df.loc[:,required+groupby].dropna(subset=['phenotype_label']))
        #return df
        self._cells = df.copy()
        self._cells.to_hdf(self.h5path,self.location+'/cells',
                              mode='r+',format='table',complib='zlib',complevel=9)
