def clustering(df, show = True, method = 'DBSCAN', \
                features = ["lat_rad", "lon_rad"], params = None, func = np.max,\
                selection = 0):
    '''
    df - input dataframe
    show - verbose param
    method - {DBSCAN, affinity, gmm, hdbscan}
    features - columns to check
    params - algorithm params space
    func - {np.min, np.max}
    selection - {0: number of clusters, 1: silhouette coefficient (small dataset)}
    '''
    from sklearn import metrics
    import matplotlib.pyplot as plt
    MAX_CLUSTERS = df.shape[0] - 1
    if method == 'DBSCAN':
        from sklearn.cluster import DBSCAN
        metric = 'haversine'
        algorithm = 'ball_tree'
        if params is None:
            params = {'meters':list(np.arange(0.001,0.2,step = 0.025)), 'min_samples':list(range(1,2,1))}
        selection_criteria = np.zeros((len(params['meters']), len(params['min_samples'])))
        columns = []
        threads = {'1':1, 'full_speed':-1}
        for i, _ in tqdm(enumerate(params['meters'])):
            for j, _ in enumerate(params['min_samples']):
                m = params['meters'][i]
                n = params['min_samples'][j]
                PREC_CLUSTERS = (m/6371)
                clusterer = DBSCAN(eps = (PREC_CLUSTERS),n_jobs = threads['full_speed'], \
                    min_samples = n,metric = metric, algorithm=algorithm)
                # Fit to our data
                X = df[features]
                clusterer.fit(X)
                size_clusters = len(np.unique(clusterer.labels_))
                if selection == 0:
                    selection_criteria[i,j] = size_clusters
                elif selection == 1:
                    if (size_clusters > 1) and (size_clusters < MAX_CLUSTERS):
                        selection_criteria[i,j] = metrics.silhouette_score(X, clusterer.labels_)
                    else:
                        selection_criteria[i,j] = -1.
                col_name = 'cluster_meters'+str(m)+'_minSamples_'+str(n)
                columns.append(col_name)
                df[col_name] = clusterer.labels_
            if show == True:
                # Data for a three-dimensional line
                m,n = selection_criteria.shape
                x = list(range(m*n))
                y = selection_criteria.flatten()
                if selection == 0:
                    title = 'Number of clusters'
                elif selection == 1:
                    title = 'Silhouette coefficient'
        flatten_clusters = selection_criteria.flatten()
        idx = list(np.where(func(flatten_clusters)== flatten_clusters)[0])[0]
        col = columns[idx]
        print('Number of clusters: ', len(df[col].unique()) - 1)
        print('Noise estimation: ', list(df[col]).count(-1)*100/df.shape[0],'%')
        if (len(df[col].unique()) - 1 > 1) and (len(df[col].unique()) - 1 < MAX_CLUSTERS):
            print('Silhouette Coefficient: %0.3f' % metrics.silhouette_score(X, df[col].values))
    elif method == 'gmm':
        from sklearn.mixture import GaussianMixture
        offset = df.shape[0]
        params = {'n_components':list(np.arange(0.1*offset,0.8*offset,step = 0.1*offset))}
        selection_criteria = np.zeros((len(params['n_components']),1))
        print(len(params['n_components']))
        columns = []
        for i, _ in tqdm(enumerate(params['n_components'])):
            m = int(np.ceil(params['n_components'][i]))
            clusterer = GaussianMixture(n_components= m,random_state = SEED)
            # Fit to our data
            X = df[features]
            clusterer.fit(X)
            labels = clusterer.predict(X)
            size_clusters = len(np.unique(labels))
            if selection == 0:
                selection_criteria[i,1] = size_clusters
            elif selection == 1:
                if (size_clusters > 1) and (size_clusters < MAX_CLUSTERS):
                    selection_criteria[i,1] = metrics.silhouette_score(X, clusterer.labels_)
                else:
                    selection_criteria[i,1] = -1.
            col_name = 'cluster_nComponents_'+str(m)
            columns.append(col_name)
            df[col_name] = labels
            if show == True:
                # Data for a three-dimensional line
                m,n = selection_criteria.shape
                x = list(range(m*n))
                y = selection_criteria.flatten()
                if selection == 0:
                    title = 'Number of clusters'
                elif selection == 1:
                    title = 'Silhouette coefficient'
        flatten_clusters = selection_criteria.flatten()
        print(selection_criteria)
        idx = list(np.where(func(flatten_clusters)== flatten_clusters)[0])[0]
        col = columns[idx]
        print('Number of clusters: ', len(df[col].unique()) - 1)
        print('Noise estimation: ', list(df[col]).count(-1)*100/df.shape[0],'%')
        if (len(df[col].unique()) - 1 > 1) and (len(df[col].unique()) - 1 < MAX_CLUSTERS):
            print('Silhouette Coefficient: %0.3f' % metrics.silhouette_score(X, df[col].values))
    elif method == 'affinity':
        from sklearn.cluster import AffinityPropagation
        if params is None:
            params = {'dampings':list(np.arange(0.7,0.98,0.025))}
        silhouette_coef, params_conv = [], []
        X = df[features]
        for damping in tqdm(params['dampings']):
            model = AffinityPropagation(damping=damping, random_state=SEED).fit(X)
            y_hat = model.predict(X)
            df['cluster_damping_'+str(damping)] = y_hat
            size_clusters = len(np.unique(y_hat))
            if  (size_clusters > 1) and (size_clusters < MAX_CLUSTERS):
                    params_conv.append(damping)
                    silhouette_coef.append(metrics.silhouette_score(X, y_hat))
        idx = list(np.where(np.max(silhouette_coef)== silhouette_coef)[0])[0]
        damping = params_conv[idx]
        col = 'cluster_damping_'+str(damping)
        print('Number of clusters: ', len(df[col].unique()) - 1)
        print('Noise estimation: ', list(df[col]).count(-1)*100/df.shape[0],'%')
        if (len(df[col].unique()) - 1 > 1) and (len(df[col].unique()) - 1 < MAX_CLUSTERS):
            print('Silhouette Coefficient: %0.3f' % metrics.silhouette_score(X, df[col].values))
        if show == True:
                # Data for a three-dimensional line
                x = list(range(len(params_conv)))
                y = silhouette_coef
                title = 'Silhouette coefficients'
    elif method == 'hdbscan':
        import hdbscan
        if params is None:
            params = {'min_samples':list(range(1,4))}
        selection_criteria, params_conv = [], []
        X = df[features]
        for min_samples in tqdm(params['min_samples']):
            clusterer = hdbscan.HDBSCAN(min_samples=min_samples, metric = 'haversine').fit(X)
            y_hat = clusterer.labels_
            df['cluster_min_samples_'+str(min_samples)] = y_hat
            size_clusters = len(np.unique(y_hat))
            selection_criteria.append(size_clusters)
            params_conv.append(min_samples)
            '''if  (size_clusters > 1) and (size_clusters < MAX_CLUSTERS):
                    params_conv.append(min_samples)
                    silhouette_coef.append(metrics.silhouette_score(X, y_hat))'''
        idx = list(np.where(func(selection_criteria)== selection_criteria)[0])[0]
        min_samples = params_conv[idx]
        col = 'cluster_min_samples_'+str(min_samples)
        print('Number of clusters: ', len(df[col].unique()) - 1)
        print('Noise estimation: ', list(df[col]).count(-1)*100/df.shape[0],'%')
        if (len(df[col].unique()) - 1 > 1) and (len(df[col].unique()) - 1 < MAX_CLUSTERS):
            print('Silhouette Coefficient: %0.3f' % metrics.silhouette_score(X, df[col].values))
        if show == True:
                # Data for a three-dimensional line
                x = list(range(len(params_conv)))
                y = selection_criteria
                title = 'Silhouette coefficients'

    # We want to keep the "noise" as reliable
    noise_size = (df.loc[df[col] == -1]).shape[0]
    clusters = df.shape[0]
    flags = list(range(clusters, clusters + noise_size))
    df.loc[df[col] == -1, col] = flags
    # Aggregate
    agg = df.groupby(col).agg({'lon':'median', 'lat':'median'}).reset_index()
    agg[features[0]], agg[features[1]] = np.radians(agg.lat), np.radians(agg.lon)
    agg.rename(columns = {col:'cluster_name'}, inplace = True)
    col = 'cluster_name'
    if show == True:
        plt.figure(figsize = (5,5))
        plt.title(title)
        plt.plot(x,y)
        plt.show()
        # Map
        '''X_real = agg[['lat','lon']].values
        yhat = agg[col].values
        _, ax = plt.subplots(figsize = (5,5))
        for cluster in np.unique(agg[col]):
            if cluster == -1:
                continue
            row_ix = np.where(yhat == cluster)
            plt.scatter(X_real[row_ix, 1], X_real[row_ix, 0])
        contextily.add_basemap(
            ax = ax,
            crs="EPSG:4326",
            source=contextily.providers.CartoDB.Positron)
        plt.show()'''

    # Matrix definition
    cols = ['lon','lat']
    X,Y,Z = _to_cartesian_df(agg[cols]*np.pi/180)
    return agg,[X,Y,Z]