# This is based on code here: https://github.com/andrew-weisman/target_classification
# Feel free to ask me (Andrew) any questions about this

def plot_unsupervised_learning_results(unsupervised_learning_results, labels):
	'''
	Plot results from unsupervised learning analyses such as PCA or tSNE, overplotting the color/style of the markers based on known labels.
	'''

	# Import relevant libraries
	import seaborn as sns
	import matplotlib.pyplot as plt
	import matplotlib.lines as mpl_lines

	# Get a reasonable set of markers and color palette
	markers = mpl_lines.Line2D.filled_markers
	nclasses = len(set(labels))
	marker_list = (markers * int(nclasses/len(markers)+1))[:nclasses]
	color_palette = sns.color_palette("hls", nclasses)

	# Plot results
	_, ax = plt.subplots(figsize=(12,7.5), facecolor='w')
	ax = sns.scatterplot(x=unsupervised_learning_results[:,0], y=unsupervised_learning_results[:,1], hue=labels, style=labels, palette=color_palette, legend='full', alpha=1, markers=marker_list, edgecolor='k', ax=ax)
	ax.legend(bbox_to_anchor=(1,1))

	return(ax)


def run_and_plot_pca_and_tsne(X, y, top_n_features=500):
	'''
	Run PCA and tSNE analyses on data X and plot the results with marker color/style by known labels y.
	Note that tSNE takes long if using many features, so for tSNE we use the top-top_n_features-variance genes.
	'''

	# Sort the features by decreasing variance
	X_by_variance = X.iloc[:,[int(x) for x in X.var().sort_values(ascending=False).index]]

	# Perform the PCA using scikit-learn
	import sklearn.decomposition as sk_decomp
	pca = sk_decomp.PCA(n_components=10)
	pca_results = pca.fit_transform(X_by_variance)
	print('Top {} PCA explained variance ratios: {}'.format(10, pca.explained_variance_ratio_))
	ax = plot_unsupervised_learning_results(pca_results, y)
	ax.set_title('PCA')

	# Perform tSNE using scikit-learn
	import sklearn.manifold as sk_manif
	tsne = sk_manif.TSNE(n_components=2)
	tsne_results = tsne.fit_transform(X_by_variance.iloc[:,:top_n_features])
	ax = plot_unsupervised_learning_results(tsne_results, y)
	ax.set_title('tSNE (using top-{}-variance features)'.format(top_n_features))
