# import matplotlib
# matplotlib.use('TkAgg')
from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from itertools import chain, combinations
from functools import partial
from matplotlib.patches import Rectangle, Circle, Wedge
import math
from matplotlib.collections import PatchCollection
import matplotlib as mpl








class FS_Plot():

	def __init__(self, rows, cols, in_sets_list, out_sets_list, wt, fs_names, bar_plot_file_path, set_row_map, fs, item_set_RID, n_bins, title, data_path, X_test, indx_set_matched_all, indx_set_matched_X_val, score_range, model_score):
	        """
	        Generates figures and axes.

	        :param rows: The number of rows of the intersection matrix

	        :param cols: The number of columns of the intersection matrix

	        :param additional_plots: list of dictionaries as specified in plot()

	        :param query: list of tuples as specified in plot()
	        """

	        self.indx_set_matched_X_val=indx_set_matched_X_val
	        self.indx_set_matched_all=indx_set_matched_all
	        self.X_test=X_test
	        self.data_path=data_path
	       
	        self.fontsize=14
	        
	        self.fs=fs
	        self.item_set_RID=item_set_RID

	        # set standard colors
	        self.greys = plt.cm.Greys([.22, 0.5, .8])
	        self.radius=0.3	
	        self.n_bins=n_bins      

	        # set figure properties
	        self.rows = rows
	        self.cols = cols
	        if(rows<=5):
	        	self.mul_factor=3 #multification factor for grid size of subplot.
	        else:
	        	self.mul_factor=1
	        self.in_sets_list=in_sets_list
	        self.out_sets_list=out_sets_list


	        self.y_max=float(self.rows/2) # Divivded by two because of half circle.
	        self.y_vals=np.arange(0,self.y_max+0.5,0.5)


	        self.x_values, self.y_values = (np.arange(cols) + 1), (np.arange(rows) + 1)	     
	        self.fig, self.ax_intbars, self.ax_intmatrix, self.ax_tablenames, self.ax_intbars2, self.ax_colorbar, self.ax_pointer = self._prepare_figure()	     
	        xlim=(0,len(self.x_values)+1)
	        ylim=(0,len(self.y_values)+1) 	        
	        self.ylim_bar=np.max(np.absolute(wt))+1.0




	        self._table_names_plot(fs_names, ylim)
	        xlim2=self._inters_sizes_plot(in_sets_list, wt, xlim)	            
	        self._inters_matrix(in_sets_list, out_sets_list, xlim, ylim, set_row_map)
	        xlim3=self._inters_sizes_plot_inverted(in_sets_list, wt, xlim)  


	        ### plot the colorbar ###
	        self.ax_pointer.set_xlim(score_range)
	        # self.ax_pointer.set_xlim((-3.0, +3.0))
	        # self.ax_pointer.set_ylim((0,2))

	        self.ax_colorbar.set_xlim(score_range)
	        self.ax_colorbar.set_ylim((0,2))

	        cmap = mpl.cm.cool
	        norm = mpl.colors.Normalize(vmin=score_range[0], vmax=score_range[1])
	        # norm = mpl.colors.Normalize(vmin=-3, vmax=+3)
	        cb1 = mpl.colorbar.ColorbarBase(self.ax_colorbar, cmap=cmap, norm=norm, orientation='horizontal')
	        # cb1.set_label('Model Score')
	        # cb1.ax.plot(0.5, 1, 'w')
	        # self.ax_colorbar.add_patch(Wedge((5.0, 0.0), 2.0, 75, 105, color='r', ec="none"))
	        self.ax_pointer.add_patch(Wedge((model_score, 0.0), 2.0, 85, 95, color='C1', ec="none"))
	        # self._strip_axes(self.ax_colorbar)
	        self._strip_axes(self.ax_pointer)





	        
	        ax=self.fig.get_axes()[2]
	        # build a rectangle in axes coords
	        left= 0.25
	        width=0.5
	        bottom=0.25
	        height=0.5
	        right = - 0.1
	        top = bottom + height	       
	       
	        ax.text(-0.05, self.fig.get_axes()[2].get_ylim()[0], 'Rule \n Importance', horizontalalignment='center',
	        fontsize=self.fontsize, verticalalignment='center', rotation='vertical', transform=ax.transAxes) # for 'figure_RID_836_2_d_0.5.pdf'
	        
	        # # # title='Features associated with top 10 rules'
	        self.fig.suptitle(title, fontsize=self.fontsize)
	        plt.show()
	       
	        # self.fig.savefig(bar_plot_file_path)	        
	        # plt.close()

	def _strip_axes(self, ax, keep_spines=None, keep_ticklabels=None):
	        """
	        Removes spines and tick labels from ax, except those specified by the user.

	        :param ax: Axes on which to operate.
	        :param keep_spines: Names of spines to keep.
	        :param keep_ticklabels: Names of tick labels to keep.

	        Possible names are 'left'|'right'|'top'|'bottom'.
	        """
	        tick_params_dict = {'which': 'both',
	                            'bottom': 'off',
	                            'top': 'off',
	                            'left': 'off',
	                            'right': 'off',
	                            'labelbottom': 'off',
	                            'labeltop': 'off',
	                            'labelleft': 'off',
	                            'labelright': 'off'}
	        if keep_ticklabels is None:
	            keep_ticklabels = []
	        if keep_spines is None:
	            keep_spines = []
	        lab_keys = [(k, "".join(["label", k])) for k in keep_ticklabels]
	        for k in lab_keys:
	            tick_params_dict[k[0]] = 'on'
	            tick_params_dict[k[1]] = 'on'
	        ax.tick_params(**tick_params_dict)
	        for sname, spine in ax.spines.items():
	            if sname not in keep_spines:
	                spine.set_visible(False)

	def _prepare_figure(self):
	        """
	        Prepares the figure, axes (and their grid) taking into account the additional plots.

	        :param additional_plots: list of dictionaries as specified in plot()
	        :return: references to the newly created figure and axes
	        """

	        # if(self.rows>=self.cols):
	        # 	scale_factor=float(self.rows/self.cols)
	        # else:
	        # 	scale_factor=float(self.cols/self.rows)

	        col_factor=float(self.rows/2) # Here divided by '2' is for half circle plot (for each feature).	  
	        scale_factor=float(col_factor/self.cols)
	        
	        w=14 # set the width of the figure
	    
	        h1=scale_factor*w # determine the height of 'intmatrix'
	        # h2=h1 # consider h1=h/2, h2=h/2
	        # h2=h1/2 # consider h1=2h/3, h2=h/3
	        # h2=h1/3	
	        h2=h1/3
	       
	        h=h1+h2
	        fig = plt.figure(figsize=(w,h))	  
	                 
	        topgs = gridspec.GridSpec(1, 1)[0, 0]
	       
	        # mul_factor=self.mul_factor	        
	        mul_factor=8 
	        intmatrix_rows=self.rows*mul_factor
	        bar_rows=int(intmatrix_rows/mul_factor)
	       
	        fig_cols = self.cols + 2
	        buffer_rows=int(self.rows*0.4)
	        # buffer_rows=int(self.rows*1.5)
	       
	        color_bar_rows=pointer_rows=4
	        
	        fig_rows = intmatrix_rows + 3*buffer_rows + bar_rows + bar_rows + color_bar_rows + pointer_rows # Extra set of rows for negative y-axis bar plot

	        
	        gs_top = gridspec.GridSpecFromSubplotSpec(fig_rows, fig_cols, subplot_spec=topgs)	        
	        

	        # tablesize_w, tablesize_h = 1, intmatrix_rows
	        tablesize_w, tablesize_h = 2, intmatrix_rows
	        intmatrix_w, intmatrix_h = tablesize_w + self.cols, intmatrix_rows
	        intbars_w, intbars_h = tablesize_w + self.cols,  bar_rows

	        
	        ax_pointer= plt.subplot(gs_top[: pointer_rows, tablesize_w :intbars_w -1])
	        ax_colorbar= plt.subplot(gs_top[pointer_rows: pointer_rows + color_bar_rows, tablesize_w :intbars_w -1])

	        ax_intbars = plt.subplot(gs_top[pointer_rows + color_bar_rows + 2*buffer_rows : pointer_rows + color_bar_rows + 2*buffer_rows + bar_rows , tablesize_w:intbars_w])		# Axis for bar graph
	        ax_intbars2 = plt.subplot(gs_top[pointer_rows + color_bar_rows + 2*buffer_rows + bar_rows : pointer_rows + color_bar_rows + 2*buffer_rows + 2*bar_rows, tablesize_w:intbars_w])		# Axis for inverted bar graph

	        ax_tablenames = plt.subplot(gs_top[pointer_rows + color_bar_rows + 2*buffer_rows + 2*bar_rows + buffer_rows:pointer_rows + color_bar_rows + 2*buffer_rows + 2*bar_rows + buffer_rows + intmatrix_rows, 0:tablesize_w])  			# Axis for feature names
	        ax_intmatrix = plt.subplot(gs_top[pointer_rows + color_bar_rows + 2*buffer_rows + 2*bar_rows + buffer_rows: pointer_rows + color_bar_rows + 2*buffer_rows + 2*bar_rows + buffer_rows + intmatrix_rows, tablesize_w:intmatrix_w])	# Axis for intersection matrix
	        
	        


	        plt.subplots_adjust(left=-0.07, bottom=0.01, right=1.02, top=0.92,
	                wspace=0.01, hspace=0.01) # for 'figure_RID_836_2_d_0.5.pdf'


	    
	        return fig, ax_intbars, ax_intmatrix, ax_tablenames, ax_intbars2, ax_colorbar, ax_pointer



	def _inters_sizes_plot(self, ordered_in_sets, inters_sizes, xlim):
	        """
	        Plots bar plot for intersection sizes.
	        :param ordered_in_sets: array of tuples. Each tuple represents an intersection. The array is sorted according
	        to the user's directives

	        :param inters_sizes: array of floats. Sorted, likewise. Feature weights in our case. Should be normalized between 0 - 1. Can be 
	        both positive and negatives. 

	        :return: Axes
	        """
	        ax = self.ax_intbars
	        ax.set_xlim(xlim)
	        ax.set_ylim(0,self.ylim_bar)
	        # width = self.radius*2
	        width = 0.3        
	        self._strip_axes(ax, keep_spines=['left'])
	        

	        indx=np.where(inters_sizes>=0)[0]
	        if(len(indx)>0):
	        	# bar_bottom_left = self.x_values - width / 2
		        bar_bottom_left = self.x_values[indx] 
		        # bar_colors = np.tile(self.greys[2], (len(ordered_in_sets), 1))
		        blue=np.array([ 0.0,  0.0,  1.0,  1.0])
		        bar_colors = np.tile(blue, (len(ordered_in_sets), 1))
		        ax.bar(bar_bottom_left, np.absolute(inters_sizes[indx]), width=width, color=bar_colors, linewidth=0)

	        ylim = ax.get_ylim()
	        label_vertical_gap = (ylim[1] - ylim[0]) / 60

	        if(len(indx)>0):	        
		        for x, y in zip(self.x_values[indx], inters_sizes[indx]):
		            ax.text(x, np.absolute(y) + label_vertical_gap, "%.2g" % y,
		                    rotation=90, ha='center', va='bottom', fontsize=self.fontsize)

	       
	        gap = max(ylim) / 500.0 * 20
	        # ax.set_ylim(ylim[0] - gap, ylim[1] + gap)
	        ylim = ax.get_ylim()
	        ax.spines['left'].set_bounds(ylim[0], ylim[1])

	        ax.yaxis.grid(True, lw=.25, color='grey', ls=':')
	        ax.set_axisbelow(True)
	       
	        return ax.get_xlim()


	def _inters_sizes_plot_inverted(self, ordered_in_sets, inters_sizes, xlim):
	        """
	        Plots bar plot for intersection sizes.
	        :param ordered_in_sets: array of tuples. Each tuple represents an intersection. The array is sorted according
	        to the user's directives

	        :param inters_sizes: array of floats. Sorted, likewise. Feature weights in our case. Should be normalized between 0 - 1. Can be 
	        both positive and negatives. 

	        :return: Axes
	        """
	        ax = self.ax_intbars2
	        
	        ax.set_xlim(xlim)
	        ax.set_ylim(0,self.ylim_bar)
	        # width = self.radius*2
	        width = 0.3
	        # self._strip_axes(ax, keep_spines=['left'], keep_ticklabels=['left'])
	        self._strip_axes(ax, keep_spines=['left'])

	        indx=np.where(inters_sizes<0)[0]

	        if(len(indx)>0):	        	
		        bar_bottom_left = self.x_values[indx] 
		        # bar_colors = np.tile(self.greys[2], (len(ordered_in_sets), 1))
		        blue=np.array([ 0.0,  0.0,  1.0,  1.0])
		        bar_colors = np.tile(blue, (len(ordered_in_sets), 1))
		        ax.bar(bar_bottom_left, np.absolute(inters_sizes[indx]), width=width, color=bar_colors, linewidth=0)

	        ylim = ax.get_ylim()
	        label_vertical_gap = (ylim[1] - ylim[0]) / 60

	        if(len(indx)>0):	        	
		        for x, y in zip(self.x_values[indx], inters_sizes[indx]):
		            ax.text(x, np.absolute(y) + label_vertical_gap, "%.2g" % y,
		                    rotation=270, ha='center', va='top', fontsize=self.fontsize)
	        	

	        ylim = ax.get_ylim()
	        ax.spines['left'].set_bounds(ylim[0], ylim[1])
	        ax.yaxis.grid(True, lw=.25, color='grey', ls=':')
	        ax.set_axisbelow(True)	        
	        ax.invert_yaxis()

	        return ax.get_xlim()


	def _inters_matrix(self, ordered_in_sets, ordered_out_sets, xlims, ylims, set_row_map):
	        """
	        Plots intersection matrix.

	        :param ordered_in_sets: Array of tuples representing sets included in an intersection. Sorted according to
	        the user's directives.

	        :param ordered_out_sets: Array of tuples representing sets excluded from an intersection. Sorted likewise.

	        :param xlims: tuple. x limits for the intersection matrix plot.

	        :param ylims: tuple. y limits for the intersection matrix plot.

	        :param set_row_map: dict. Maps data frames (base sets) names to a row of the intersection matrix

	        :return: Axes
	        """
	        ax = self.ax_intmatrix
	        ax.set_xlim(xlims)
	        
	        ax.set_ylim((0,self.y_max))
	        

	        ax.invert_yaxis()

	        if len(self.y_values) > 1:
	            row_width = self.y_values[1] - self.y_values[0]
	        else:
	            row_width = self.y_values[0]

	        row_width=float(row_width)/self.mul_factor
	        
	        radius=self.radius

	        self._strip_axes(ax)

	        background = plt.cm.Greys([.09])[0]

	        # for r, y in enumerate(self.y_values):
	        #     if r % 2 == 0:	            
	        #         # ax.add_patch(Rectangle((xlims[0], self.y_vals[r]), height=float(row_width)/2,
	        #         #                        width=xlims[1], color=background, zorder=0))
	        #         ax.add_patch(Rectangle((xlims[0], self.y_vals[r]), height=0.45,
	        #                                width=xlims[1], color=background, zorder=0))
	        for r, x in enumerate(self.x_values):
	            if r % 2 == 0:	            
	                # ax.add_patch(Rectangle((xlims[0], self.y_vals[r]), height=float(row_width)/2,
	                #                        width=xlims[1], color=background, zorder=0))
	                ax.add_patch(Rectangle((self.x_values[r]-(self.radius+0.05), self.y_vals[0]), height=self.y_max,
	                                       width=(self.radius+0.05)*2, color=background, zorder=0))




	        feature_range_array=np.load(self.data_path + 'Feature_range_array.npy')
	        rule_array=np.load(self.data_path + 'Rule_array.npy')	

	        


	        for col_num, (in_sets, out_sets) in enumerate(zip(ordered_in_sets, ordered_out_sets)):
	            in_y = [set_row_map[s] for s in in_sets]
	            out_y = [set_row_map[s] for s in out_sets]	            
	           
	            self._draw_patch_out_sets(ax, col_num, self.y_vals[out_y], radius)
	            items=self.item_set_RID[col_num]
	            f_indices=[s for s in in_sets]
	            self._draw_patch_in_sets(items, f_indices, col_num, self.y_vals[in_y], feature_range_array, rule_array, ax, radius)

	        # Draw a blue rectangle highlighting the selected features.
	        blue=np.array([ 0.0,  0.0,  1.0,  1.0])
	        for i in range(len(self.indx_set_matched_all)):
	        	x_start=self.indx_set_matched_X_val[i]

	        	is_tmp = self.indx_set_matched_all[i]
	        	is_tmp_y = [set_row_map[s] for s in is_tmp]	        	
	        	y_start=min(is_tmp_y)
	        	y_stop=max(is_tmp_y) 
	        	ax.add_patch(Rectangle((self.x_values[x_start]-(self.radius+0.07), self.y_vals[y_start] - 0.5), 
	        				height=self.y_vals[y_stop] - (self.y_vals[y_start] - 0.5), width=(self.radius+0.07)*2, color=blue, zorder=5, fill=False))

	        	# as we subtracted 0.5 from y_start, we need to consider the same while calculating the heights.
	           
	            

	def _table_names_plot(self, sorted_set_names, ylim):
	        ax = self.ax_tablenames
	        # ax.set_ylim(ylim)
	        ax.set_ylim((0,self.y_max))
	        
	        xlim = ax.get_xlim()
	        tr = ax.transData.transform
	        ax.invert_yaxis()

	        # y_step_mid=0.25
	        
	        for i, name in enumerate(sorted_set_names):
	            ax.text(x=1,  # (xlim[1]-xlim[0]/2),
	                    y=self.y_vals[i] + 0.25,
	                    s=name,
	                    # horizontalalignment='right',verticalalignment='center', rotation='vertical',fontweight='bold', 
	                    fontsize=self.fontsize,
	                    fontweight='bold',
	                    clip_on=True,
	                    va='center',
	                    ha='right',
	                    transform=ax.transData,
	                    family='monospace')	        
	        ax.axis('off')






	def _draw_patch_out_sets(self, ax, col_num, out_y, radius):
		for j in range(len(out_y)):
			# ax.add_patch(Wedge((self.x_values[col_num], out_y[j]-0.1), radius, 180, 360, color=self.greys[0], ec="none"))
			ax.add_patch(Wedge((self.x_values[col_num], out_y[j]-0.1), radius, 180, 360, color='g', alpha=0.2, ec="none"))





	def _draw_patch_in_sets(self, items, f_indices,  col_num, y, feature_range_array, rule_array, ax, radius):
		patches = []
		rule=[]
		feature_range=[]
		items=list(map(int, items))
		for j in range(len(items)):		
			Min=float(feature_range_array[f_indices[j],1])
			Max=float(feature_range_array[f_indices[j],2])
			Range=Max-Min

			rule.append(rule_array[items[j]])
			feature_range.append(feature_range_array[f_indices[j]])
			
			rule_min=float(rule_array[items[j], 1])
			rule_max=float(rule_array[items[j], 2]) # For Overlapping rules
			angle=(180*(rule_min - Min))/Range

			# bin_size=float(Range/self.n_bins)   # For equal step size rules
			bin_size=rule_max - rule_min		# For Overlapping rules
			base=180
			start_angle=base+angle
			end_angle=start_angle + 180*(bin_size/Range)

			v_gap=0.1


			### patch for individual feature value ###
			feature_val=self.X_test[0,f_indices[j]]
			angle_feature=base + (180*(feature_val- Min))/Range
			x1=radius*np.cos((np.pi/180)*angle_feature) # Input angle is in radian
			y1=radius*np.sin((np.pi/180)*angle_feature) # Input angle is in radian
			r1=float(radius/3)
			angle_feature_start= angle_feature - 180*(bin_size/Range)
			angle_feature_end= angle_feature + 180*(bin_size/Range)


		
			# ax.add_patch(Wedge((self.x_values[col_num], y[j] - v_gap), radius, 180, 360, color=self.greys[0], ec="none"))
			# ax.add_patch(Wedge((self.x_values[col_num], y[j] - v_gap), radius, 180, 345, color=self.greys[1], ec="none"))
			# ax.add_patch(Wedge((self.x_values[col_num], y[j] - v_gap), radius, start_angle, end_angle, color=self.greys[2], ec="none"))

			ax.add_patch(Wedge((self.x_values[col_num], y[j] - v_gap), radius, 180, 360, color='r', alpha=0.2, ec="none"))
			# ax.add_patch(Wedge((self.x_values[col_num], y[j] - v_gap), radius, 180, 345, color='r', alpha=0.4, ec="none"))
			ax.add_patch(Wedge((self.x_values[col_num], y[j] - v_gap), radius, start_angle, end_angle, color='r', alpha=1.0, ec="none"))

			ax.add_patch(Wedge((self.x_values[col_num] + x1, y[j] - v_gap + y1), r1, angle_feature_start, angle_feature_end, color='C1', alpha=1.0, ec="none"))


		

















