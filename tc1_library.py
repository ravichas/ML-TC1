# This is based on code here: https://github.com/andrew-weisman/target_classification
# Feel free to contact (Andrew) any questions about this

def plot_results(results_to_plot, MYLABELS):

	# Import relevant libraries
	import seaborn as sns
	import matplotlib.pyplot as plt
	import matplotlib.lines as mpl_lines

	# Get a reasonable set of markers and color palette
	markers = mpl_lines.Line2D.filled_markers
	nclasses = len(set(MYLABELS))
	marker_list = (markers * int(nclasses/len(markers)+1))[:nclasses]
	color_palette = sns.color_palette("hls", nclasses)

	# Plot results
	_, ax = plt.subplots(figsize=(12,7.5), facecolor='w')
	ax = sns.scatterplot(x=results_to_plot[:,0], y=results_to_plot[:,1], hue=MYLABELS, style=MYLABELS, palette=color_palette, legend='full', alpha=1, markers=marker_list, edgecolor='k', ax=ax)
	ax.legend(bbox_to_anchor=(1,1))

	return(ax)
