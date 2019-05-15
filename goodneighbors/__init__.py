import h5py, json, sys, math
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from goodneighbors.plots import plot_counts_per_radius, plot_kmeans_evaluation, plot_cluster_composition
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import leaves_list
from scipy.cluster.hierarchy import linkage
from sklearn.preprocessing import scale
from scipy.spatial.distance import cdist

class GoodNeighbors(object):
    """
    Class for executing and storing the neighborhood analysis

    Properties:
        :microns_per_pixel: (int) set in initialization
        :radius: (float) pixel size of radius, set by set_neighborhood_fractions via the radius_um (in microns)
        :groupby: (list) of cells fields who as a group can identify image frames
        :h5path: (str) location of the h5path storing things
        :location: (str) subdirectory in the h5object

    """
    def __init__(self,h5path,location='',mode='r',microns_per_pixel=np.nan,verbose=False):
        self.h5path = h5path
        self.mode = mode
        self.location = location
        self._cells = None
        self.verbose = verbose
        if mode in ['w','r+','a']:
            f = h5py.File(self.h5path,mode)
            if mode in ['r+','a'] and location+'/cells' in f:
                f.close()
                return  # we've already made this                
            if location != '': f.create_group(location+'/cells')
            dset = f.create_dataset('/meta', (100,), dtype=h5py.special_dtype(vlen=str))
            dset.attrs['microns_per_pixel'] = microns_per_pixel
            dset.attrs['groupby'] = json.dumps([])
            dset.attrs['radius'] = np.nan
            dset.attrs['scaler'] = "None"
            #dset.attrs['id'] = uuid4().hex
            f.close()
    def label_clusters(self,d):
        if not self.mode in ['w','r+','a']: raise ValueError('cant write for readonly')
        c = self.clusters.copy()
        d2 = {}
        for k,vs in d.items():
            for v in vs: 
                if v in d2: raise ValueError("Name can only be defined once")
                d2[v] = k
        if set(d2.keys()) != set(c['cluster_id']): raise ValuError("must define all clusters")
        c['cluster_name'] = c.apply(lambda x: d2[x['cluster_id']],1)
        if self.tsne is not None:
            t = self.tsne.copy()
            t['cluster_name'] = t.apply(lambda x: d2[x['cluster_id']],1)
            self.tsne = t
            #t.to_hdf(self.h5path,self.location+'/cells/tsne',
            #         mode='r+',format='table',complib='zlib',complevel=9)
        self.clusters = c
        #c.to_hdf(self.h5path,self.location+'/cells/clusters',
        #             mode='r+',format='table',complib='zlib',complevel=9)

    def get_cluster_composition(self,label='cluster_id',autosort=True):
        clusters = self.clusters
        counts = _get_fractions(clusters,label=label)
        return counts, plot_cluster_composition(counts,label,autosort)

    #@property
    #def tsne(self):
    #    return pd.read_hdf(self.h5path,self.location+'/cells/tsne')
    def calculate_TSNE(self,indecies=None,sample=5000,n_jobs=1,multicore=False,**kwargs):
        """
        Calculate the TSNE and store it in the h5 object

        Args:
            indecies (list): select a subset based on indecies
            sample (int): number of cells to downsample to if None use them all
            n_jobs (int): number of cpus to use (if multicore is True)
            multicore (bool): use the MultiCoreTSNE package
            **kwargs: pass any other arguments to TSNE
        """
        if not self.mode in ['w','r+','a']: raise ValueError('cant write for readonly')
        dsdata = self.fractions.copy()
        if indecies is not None:
            dsdata = dsdata.loc[indecies]
        if sample is not None:
            if dsdata.shape[0] < sample: sample = dsdata.shape[0]
            dsdata = dsdata.sample(n=sample)
        if self.verbose: sys.stderr.write("executing TSNE decomposition on "+str(dsdata.shape[0])+" cells\n")
        tsne = None
        if multicore: 
            from MulticoreTSNE import MulticoreTSNE as mTSNE
            if self.verbose: sys.stderr.write("Using MulticoreTSNE\n")
            tsne = mTSNE(n_jobs=n_jobs,**kwargs).fit_transform(dsdata)
        else:
            if self.verbose: sys.stderr.write("Using sklearn TSNE\n")
            tsne = TSNE(**kwargs).fit_transform(dsdata)
        tsne  = pd.DataFrame(tsne,columns=['x','y'])
        tsne.index = dsdata.index
        tsne = tsne.reset_index().merge(self.clusters.reset_index()[['cluster_id','k','db_id']+self.groupby],
                                on=['db_id'])
        tsne['cluster_name'] = tsne['cluster_id'].astype(str)
        self.tsne = tsne
        #tsne.to_hdf(self.h5path,self.location+'/cells/tsne',
        #             mode='r+',format='table',complib='zlib',complevel=9)
        #return tsne

    #@property
    #def clusters(self):
    #    return pd.read_hdf(self.h5path,self.location+'/cells/clusters')

    def calculate_clusters(self,k):
        if not self.mode in ['w','r+','a']: raise ValueError('cant write for readonly')
        km, clusters = self._cluster_kmeans(k)
        clusters['k'] = k
        clusters = clusters.merge(self.cells,on=['db_id'])
        clusters = clusters.set_index('db_id')
        ## now reoder the clusters
        mat = _get_fractions(clusters,label='cluster_id').\
            pivot(columns='phenotype_label',index='cluster_id',values='fraction')
        lorder = list(leaves_list(linkage(mat,method='ward',optimal_ordering=True)))
        lorder = mat.iloc[lorder].index
        d = dict([(x,i) for i,x in enumerate(lorder)])
        clusters['cluster_id'] = clusters['cluster_id'].apply(lambda x: d[x])

        clusters['cluster_name'] = clusters['cluster_id'].astype(str)
        self.clusters= clusters
        return km, d
        #clusters.to_hdf(self.h5path,self.location+'/cells/clusters',
        #             mode='r+',format='table',complib='zlib',complevel=9)


    def _cluster_kmeans(self,k):
        fmat = self.fractions
        km = KMeans(n_clusters=k).fit(fmat)
        cids = km.labels_
        clusters = pd.DataFrame({'cluster_id':cids},index=fmat.index).reset_index()
        km, clusters
        return  km, clusters

    def get_kmeans_evaluation(self,kmin=2,kmax=20):
        kdata = []
        for k in range(kmin,kmax):
            km, clusters = self._cluster_kmeans(k)
            # how how much of the population does the smallest cluster account for
            accounted = clusters.groupby('cluster_id')[['db_id']].count().\
                rename(columns={'db_id':'count'}).reset_index()
            smallest = accounted.apply(lambda x: x['count']/clusters.shape[0],1).min()
            kdata.append([k,km.inertia_,smallest])
        kdata = pd.DataFrame(kdata,columns=['k','inertia','smallest_cluster_fraction'])
        return kdata, plot_kmeans_evaluation(kdata)


    #def neighborhood_fraction_matrix(self):
    #    index_labels = list(set(list(self.neighborhood_fractions.columns))-set(['n_phenotype_label','fraction']))
    #    fmat = self.neighborhood_fractions.set_index(index_labels).pivot(columns='n_phenotype_label')
    #    if self.standardize:
    #        cnames = fmat.columns
    #        rnames = fmat.index
    #        fmat = pd.DataFrame(scale(fmat),columns=cnames,index=rnames)
    #    return fmat

    #@property
    #def neighborhood_fractions(self):
    #    return pd.read_hdf(self.h5path,self.location+'/cells/fractions')

    #def set_neighborhood_fractions(self,radius_um=None,radius_pixels=None):
    #    if radius_um is None and radius_pixels is None: raise ValueError('set one of the radius distances')
    #    if radius_um is not None and radius_pixels is not None: raise ValueError('only set one distance')
    #    if radius_um is not None:
    #        self.radius = radius_um/self.microns_per_pixel
    #    else:
    #        self.radius = radius_pixels
    #    dist = self.measure_distance(use_cache=True)
    #    focus = dist.loc[dist['distance']<radius_pixels]
    #    totals = focus.groupby(self.groupby+['phenotype_label','db_id']).count()[['x']].\
    #        rename(columns={'x':'total'}).reset_index()
    #    indiv = focus.groupby(self.groupby+['phenotype_label','db_id','n_phenotype_label']).count()[['x']].\
    #        rename(columns={'x':'count'}).reset_index()
    #    fracs = totals.merge(indiv,on=self.groupby+['phenotype_label','db_id'])
    #    fracs['fraction'] = fracs.apply(lambda x: x['count']/x['total'],1)
    #    fracs = fracs.set_index(self.groupby+['phenotype_label','db_id','total'])\
    #        [['n_phenotype_label','fraction']].\
    #        pivot(columns='n_phenotype_label')
    #    fracs.columns = fracs.columns.droplevel(0)
    #    fracs = fracs.fillna(0).stack().reset_index().rename(columns={0:'fraction'})
    #    fracs.to_hdf(self.h5path,self.location+'/cells/fractions',
    #                 mode='r+',format='table',complib='zlib',complevel=9)
    #    #fracs['radius'] = radius_pixels
    #    return 

    def calculate_neighbor_counts(self,radius,units='microns'):
        """
        count the neighbors at the radius
        """
        if units not in ('microns','pixels'): raise ValueError("must be microns or pixels for units")
        if units=='microns' and not self.microns_per_pixel: raise ValueError("cannot convert um to pixels if you dont set microns_per_pixel")
        radius = radius/self.microns_per_pixel
        if self.radius != radius or self.counts is None:
            self.counts = None
        else:
            return # we've already got a value 
        self.radius = radius
        if self.verbose: sys.stderr.write("Max distance is "+str(radius)+"\n")
        full = self.cells.copy().reset_index()
        phenotypes = pd.DataFrame({'phenotype_label':self.cells['phenotype_label'].unique()})
        phenotypes['_key'] = 1
        total = pd.DataFrame({'db_id':self.cells.index.tolist()})
        total['_key'] = 1
        total = total.merge(phenotypes,on='_key')
        ## Execute distance measure on a per-image basis
        images = full[self.groupby].drop_duplicates().copy()
        collect = []
        for i,r in images.iterrows():
            idf = pd.DataFrame(r).T
            #print(idf)
            if self.verbose: sys.stderr.write("================\n")
            if self.verbose: sys.stderr.write("Calculating distances for "+str(r)+"\n")
            one = full.merge(idf,on=self.groupby).set_index('db_id')

            # get all the distances
            coords = list(zip(one['x'],one['y']))
            dist = cdist(coords,coords)
            dist = pd.DataFrame(dist,columns=one.index,index=one.index)

            # get the distances we are interested in
            s = one.apply(lambda x: 
                dist.columns[dist.loc[x.name]<radius].tolist()
            ,1)
            s = pd.DataFrame(s).apply(lambda x: pd.Series(*x),1).stack().reset_index().\
                drop(columns='level_1').\
                rename(columns={'level_0':'db_id',0:'n_db_id'}).dropna()
            s = s.astype(int)
            s = s.loc[s['db_id']!=s['n_db_id']].set_index('n_db_id') # going to join on this neighbor id

            # Shape the counts into a matrix
            cnts = s.merge(one[['phenotype_label']],left_index=True,right_index=True).reset_index().groupby(['db_id','phenotype_label']).count().\
                reset_index().sort_values(['db_id','phenotype_label'])
            ids = pd.DataFrame({'db_id':one.index.tolist()})
            ids['_key'] = 1
            cnts = ids.merge(phenotypes,on='_key').drop(columns=['_key']).merge(cnts,on=['db_id','phenotype_label'],how='left').fillna(0)
            cnts['index'] = cnts['index'].astype(int)
            cnts = cnts.pivot(columns='phenotype_label',index='db_id',values='index')

            #subdist['_key'] = 1
            #if self.verbose: sys.stderr.write("merging to self\n")
            #subdist = subdist.merge(subdist.rename(columns={'x':'n_x','y':'n_y',
            #                                     'phenotype_label':'n_phenotype_label',
            #                                     'db_id':'n_db_id'}),on='_key').drop(columns=['_key'])
            #if self.verbose: sys.stderr.write("we have "+str(subdist.shape[0])+" to check.\n")
            #subdist = subdist.loc[subdist['db_id']!=subdist['n_db_id']]
            #xdist = subdist['x'].subtract(subdist['n_x'])
            #ydist = subdist['y'].subtract(subdist['n_y'])
            #subdist['distance'] = np.sqrt(xdist.multiply(xdist).add(ydist.multiply(ydist)))
            #subdist = subdist.loc[subdist['distance']<radius].copy()
            #if self.verbose: sys.stderr.write("got cut down version with "+str(subdist.shape[0])+"\n")
            #subdist = subdist.groupby(['db_id','n_phenotype_label']).count()[['n_db_id']].\
            #    rename(columns={'n_db_id':'count'}).\
            #    reset_index().rename(columns={'n_phenotype_label':'phenotype_label'}).\
            #    pivot(columns='phenotype_label',index='db_id',values='count').\
            #    fillna(0).astype(int).copy()
            collect.append(cnts)
        self.counts = pd.concat(collect)
        return #self._counts

    def get_counts_per_radius(self,min_radius=0,max_radius=150,step_radius=2):
        """
        
        """
        dist = self.measure_distance(use_cache=True)
        census = []
        for i in range(step_radius,min(max_radius,int(dist['distance'].max())),step_radius):
            if self.verbose: sys.stderr.write("Step "+str(i)+"\n")
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

        return census, plot_counts_per_radius(census)
        
    #def measure_distance(self,max_radius=None,use_cache=False):
    #    """
    #    Calculate the distance and store it in the cache
    #    """
    #    if self._distance is not None and use_cache : return self._distance
    #    if max_radius is None:
    #        max_radius = int(math.sqrt(self.cells['x'].max()*\
    #                                   self.cells['x'].max()+\
    #                                   self.cells['y'].max()*\
    #                                   self.cells['y'].max()))+1
    #    if self.verbose: sys.stderr.write("Max distance is "+str(max_radius)+"\n")
    #    one = self.cells.copy().reset_index()
    #    ## Execute distance measure on a per-image basis
    #    images = one[self.groupby].drop_duplicates().copy()
    #    dist = []
    #    for i,r in images.iterrows():
    #        idf = pd.DataFrame(r).T
    #        if self.verbose: sys.stderr.write("Calculating distances for "+str(r)+"\n")
    #        sub = one.merge(idf,on=self.groupby)
    #        subdist = sub.merge(sub.rename(columns={'x':'n_x','y':'n_y',
    #                                             'phenotype_label':'n_phenotype_label',
    #                                             'db_id':'n_db_id'}),on=self.groupby)
    #        subdist = subdist.loc[subdist['db_id']!=subdist['n_db_id']]
    #        xdist = subdist['x'].subtract(subdist['n_x'])
    #        ydist = subdist['y'].subtract(subdist['n_y'])
    #        subdist['distance'] = np.sqrt(xdist.multiply(xdist).add(ydist.multiply(ydist)))
    #        subdist = subdist[subdist['distance']<max_radius]
    #        dist.append(subdist)
    #    self._distance = pd.concat(dist)
    #    return self._distance

    @property
    def scaler(self):
        f = h5py.File(self.h5path,'r')
        levels = (self.location+'/meta').split('/')
        d = f
        for level in levels:
            if level == '': continue
            d = d[level]
        return d.attrs['scaler']
    @scaler.setter
    def scaler(self,value):
        if not self.mode in ['a','r+','w']: raise ValueError('cant write on readonly')
        f = h5py.File(self.h5path,'r+')
        levels =(self.location+'/meta').split('/')
        d = f
        for level in levels:
            if level == '': continue
            d = d[level]
        d.attrs['scaler'] = str(value)
        return

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
    def counts(self):
        f = h5py.File(self.h5path,'r')
        if self.location+'/cells/counts' in f:
            f.close()
            return pd.read_hdf(self.h5path,self.location+'/cells/counts')
        f.close()
        return None
    @counts.setter
    def counts(self,value):
        if not self.mode in ['a','r+','w']: raise ValueError('cant write on readonly')
        if value is None:
            f = h5py.File(self.h5path,'r+')
            if self.location+'/cells/counts' in f: del f[self.location+'/cells/counts']
            f.close()
            self.fractions = None
            return
        value.to_hdf(self.h5path,self.location+'/cells/counts',
                            mode='r+',format='table',complib='zlib',complevel=9)
        return

    @property
    def clusters(self):
        f = h5py.File(self.h5path,'r')
        if self.location+'/cells/clusters' in f:
            f.close()
            return pd.read_hdf(self.h5path,self.location+'/cells/clusters')
        f.close()
        return None
    @clusters.setter
    def clusters(self,value):
        if not self.mode in ['a','r+','w']: raise ValueError('cant write on readonly')
        if value is None:
            f = h5py.File(self.h5path,'r+')
            if self.location+'/cells/clusters' in f: del f[self.location+'/cells/clusters']
            f.close()
            return
        value.to_hdf(self.h5path,self.location+'/cells/clusters',
                            mode='r+',format='table',complib='zlib',complevel=9)
        return

    @property
    def tsne(self):
        f = h5py.File(self.h5path,'r')
        if self.location+'/cells/tsne' in f:
            f.close()
            return pd.read_hdf(self.h5path,self.location+'/cells/tsne')
        f.close()
        return None
    @tsne.setter
    def tsne(self,value):
        if not self.mode in ['a','r+','w']: raise ValueError('cant write on readonly')
        if value is None:
            f = h5py.File(self.h5path,'r+')
            if self.location+'/cells/tsne' in f: del f[self.location+'/cells/tsne']
            f.close()
            return
        value.to_hdf(self.h5path,self.location+'/cells/tsne',
                            mode='r+',format='table',complib='zlib',complevel=9)
        return

    @property
    def fractions(self):
        f = h5py.File(self.h5path,'r')
        if self.location+'/cells/fractions' in f:
            f.close()
            return pd.read_hdf(self.h5path,self.location+'/cells/fractions')
        f.close()
        return None
    @fractions.setter
    def fractions(self,value):
        if not self.mode in ['a','r+','w']: raise ValueError('cant write on readonly')
        if value is None:
            f = h5py.File(self.h5path,'r+')
            if self.location+'/cells/fractions' in f: del f[self.location+'/cells/fractions']
            f.close()
            return
        value.to_hdf(self.h5path,self.location+'/cells/fractions',
                            mode='r+',format='table',complib='zlib',complevel=9)
        return

    def calculate_fractions(self,scaler=None,fillna=0):
        # i.e. from sklearn.preprocessing import scale
        if self.counts is None: raise ValueError("Set counts before calculating fractions")
        if scaler != self.scaler: 
            self.fractions = None
        elif self.fractions is not None:
            return
        self.scaler = scaler 
        cnts = self.counts.copy()
        tot = cnts.sum(1)
        for i in range(0,cnts.shape[1]):
            cnts.iloc[:,i] = cnts.iloc[:,i].divide(tot)
        if scaler is not None:
            cnts = pd.DataFrame(scaler.fit_transform(cnts),columns=cnts.columns)
        cnts.index.name = 'db_id'
        if fillna is not None:
            cnts = cnts.fillna(fillna)
        self.fractions = cnts
    
        
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
    def standardize(self):
        f = h5py.File(self.h5path,'r')
        levels = (self.location+'/meta').split('/')
        d = f
        for level in levels:
            if level == '': continue
            d = d[level]
        return d.attrs['standardize']
    @standardize.setter
    def standardize(self,value):
        if not self.mode in ['a','r+','w']: raise ValueError('cant write on readonly')
        f = h5py.File(self.h5path,'r+')
        levels =(self.location+'/meta').split('/')
        d = f
        for level in levels:
            if level == '': continue
            d = d[level]
        d.attrs['standardize'] = value
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
        """
        Access the pandas.DataFrame stored in the h5 object and return it

        Returns:
            (pandas.DataFrame)
        """
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
        """
        Define the cells as a pandas.DataFrame

        Inputs:
            :df: (pandas.DataFrame) Must have columns defined ``['x','y','phenotype_label']`` the index must be unique
            :groupby: (list) List of fields that will group the pandas.DataFrame by image frame
        """
        if self.mode not in ['w','r+','a']: raise ValueError("Cannot set data in read-only")
        if not df.index.is_unique: raise ValueError("Cells must have a unique index")
        self.groupby = groupby
        required = ['x','y','phenotype_label']
        for r in required: 
            if r not in df.columns: raise ValuError("Cell dataframe input needs at least "+str(required))
        df = pd.DataFrame(df.loc[:,required+groupby].dropna(subset=['phenotype_label']))
        #return df
        self._cells = df.copy()
        self._cells.index.name = 'db_id'
        self._cells.to_hdf(self.h5path,self.location+'/cells',
                              mode='r+',format='table',complib='zlib',complevel=9)

def _get_fractions(clusters,label='cluster_id'):
        cnames = pd.DataFrame({label:clusters[label].unique()})
        cnames['_key'] =1
        pnames = pd.DataFrame({'phenotype_label':clusters['phenotype_label'].unique()})
        pnames['_key'] =1
        complete = cnames.merge(pnames,on='_key').drop(columns='_key')
        totals = clusters.groupby(label).count()[['x']].rename(columns={'x':'total'})
        counts = clusters.groupby([label,'phenotype_label']).count()[['x']].rename(columns={'x':'count'})
        counts = counts.reset_index().merge(totals.reset_index(),on=[label])
        counts['fraction'] = counts.apply(lambda x: x['count']/x['total'],1)
        counts = complete.merge(counts,on=[label,'phenotype_label'],how='left').fillna(0)
        return counts
