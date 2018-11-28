import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from plotly.graph_objs import *
import plotly.tools as tls
import plotly.offline as py
import csv
import numpy as np

def plot_PCA(csv_filename):
    df = pd.read_csv(
        filepath_or_buffer='/home/sachin/DV/Projects/project/intermediateData/' + str(csv_filename) + '.csv',
        sep=',')

    df.dropna(how="all", inplace=True) # drops the empty line at file-end

    X = df.ix[:,1:].values
    y = df.ix[:,0].values
    X_std = StandardScaler().fit_transform(X)

    pca = PCA(n_components=10)
    Y_sklearn = pca.fit_transform(X_std)
    traces = []
    names = y

    X_centered = X_std - np.mean(X, axis=0)
    cov_matrix = np.dot(X_centered.T, X_centered)
    eigenvalues = pca.explained_variance_
    bar_values = []
    for eigenvalue, eigenvector in zip(eigenvalues, pca.components_):    
        bar_values.append(np.dot(eigenvector.T, np.dot(cov_matrix, eigenvector)))
    with open("../intermediateData/pca_eigenvalues.csv", 'wb') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(["index", "eigenvalues"])
        i = 0
        for value in bar_values:
            wr.writerow([i, value])
            i += 1
    X = []
    Y = []
    print pca.components_
    with open("../intermediateData/pca_axes.csv", 'wb') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(["X", "Y"])
        for value in pca.components_:
            wr.writerow([0, 0])
            wr.writerow([value[0], value[1]])

    '''
    for name in names:

        trace = Scatter(
            x=Y_sklearn[y==name,0],
            y=Y_sklearn[y==name,1],
            mode='markers',
            name=name,
            marker=scatter.Marker(
                size=12,
                line=dict(
                    width = 2,
                ),
                color='rgba(255, 0, 0, 1.0)',
                opacity=1))
        traces.append(trace)

    layout = Layout(
        xaxis=dict(
            autorange=True,
            showgrid=False,
            zeroline=False,
            showline=False,
            ticks='',
            showticklabels=False
        ),
        yaxis=dict(
            autorange=True,
            showgrid=False,
            zeroline=False,
            showline=False,
            ticks='',
            showticklabels=False
        ),
         showlegend=False)
    
    fig = Figure(data=traces, layout=layout)
    py.plot(fig, filename="../intermediateData/" + str(csv_filename) + ".html", config={"displayModeBar": False, "showLink": False}, auto_open=False)
    '''

for i in range(5):
    plot_PCA(i)