import matplotlib.pyplot as plt
import seaborn as sns

# Add labels to a bar graph
def add_barplot_labels(ax, orient, fontsize):
    if orient in ['h', 'x', 'horizontal']:
        # Barplot oriented horizontally
        for p in ax.patches:
            width = p.get_width()
            if width != 0:
                ax.annotate(
                    f'{int(width)}', 
                    xy=(width, p.get_y() + p.get_height() / 2),
                    fontsize=fontsize, ha='left', va='center'
                )
        return ax    
    elif orient in ['v', 'y', 'vertical']:
        # Barplot oriented vertically
        for p in ax.patches:
            y_value = p.get_height()
            rounded_value = round(y_value)
            if rounded_value != 0:
                ax.annotate(
                    f'{int(rounded_value):d}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    fontsize=fontsize, ha='center', va='center',
                    xytext=(0, 5), textcoords='offset points'
                )    
        return ax

# Create scatter plot with/out regression line
def make_scatter_plot(
        fig, ax, x=None, y=None, y_pred=None, scat_color=None, line_color=None, 
        xlabel=None, ylabel=None, title=None, x_lim=None, y_lim=None, 
        xticks=None, add_eqns=False, lr_eqn=None, r2_eqn=None, mse_eqn=None, 
        rmse_eqn=None, eqn_loc=None, r2_loc=None, mse_loc=None,
        rmse_loc=None):
    # Title and axes labels
    if title is not None:
        ax.set_title(title, fontweight='bold', fontsize=16)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontweight='bold', fontsize=14)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontweight='bold', fontsize=14)

    # x and y limits and tickmarks
    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    if xticks.any() is not None:
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(val) for val in xticks])
                       
    # Make graph
    for i in range(len(x)):
        ax.scatter(x[i], y[i], color=scat_color[i])
        if y_pred:
            ax.plot(x[i], y_pred[i], color=line_color[i], linewidth=2)
        
        # Equations
        if add_eqns and len(x) > 1:
            if lr_eqn is not None:
                ax.text(
                    eqn_loc[i][0], eqn_loc[i][1], lr_eqn[i], 
                    color=scat_color[i], fontsize=14,math_fontfamily='cm'
                )
            if r2_eqn is not None:
                ax.text(
                    eqn_loc[i][0], r2_loc[i], r2_eqn[i], color=scat_color[i], 
                    fontsize=14, math_fontfamily='cm'
                )
            if mse_eqn is not None:
                ax.text(
                    eqn_loc[i][0], mse_loc[i], mse_eqn[i], color=scat_color[i], 
                    fontsize=14, math_fontfamily='cm'
                )                
            if rmse_eqn is not None:
                ax.text(
                    eqn_loc[i][0], rmse_loc[i], rmse_eqn[i],
                    color=scat_color[i], fontsize=14, math_fontfamily='cm'
                )                
        else:
            if lr_eqn is not None:
                ax.text(
                    eqn_loc[0], eqn_loc[1], lr_eqn[0], color=scat_color[i], 
                    fontsize=14, math_fontfamily='cm'
                )                
            if r2_eqn is not None:
                ax.text(
                    eqn_loc[0], r2_loc[0], r2_eqn[0], color=scat_color[i], 
                    fontsize=14, math_fontfamily='cm'
                )
            if mse_eqn is not None:
                ax.text(
                    eqn_loc[0], mse_loc[0], mse_eqn[0], color=scat_color[i], 
                    fontsize=14, math_fontfamily='cm'
                )
            if rmse_eqn is not None:
                ax.text(
                    eqn_loc[0], rmse_loc[0], rmse_eqn[0], color=scat_color[i], 
                    fontsize=14, math_fontfamily='cm'
                )            
                        
    return fig, ax

# Make a Histogram 
def make_histogram(
        df, x=None, hue=None, stat='count', binrange=None, bins='auto', 
        alpha=1, palette=None, xlabel=None, ylabel=None, title=None, y=None, 
        weights=None, binwidth=None, discrete=None, cumulative=False, 
        common_bins=True, common_norm=True, multiple='layer', element='bars', 
        fill=True, shrink=1, kde=False, kde_kws=None, line_kws=None, thresh=0, 
        pthresh=None, pmax=None, cbar=False, cbar_ax=None, cbar_kws=None, 
        hue_order=None, hue_norm=None, color=None, log_scale=None, legend=True, 
        hue_labels=None, ax=None, add_bar_labels=None, bar_orient='v', 
        **kwargs):
    print("="*60)
    print(f"Making a Histogram of '{x}' with '{hue}' hue\n")
    sns.histplot(
        data=df, x=x, y=y, hue=hue, weights=weights, stat=stat, bins=bins, 
        binwidth=binwidth, binrange=binrange, discrete=discrete, 
        cumulative=cumulative, common_bins=common_bins, 
        common_norm=common_norm, multiple=multiple, element=element, fill=fill, 
        shrink=shrink, kde=kde, kde_kws=kde_kws, line_kws=line_kws, 
        thresh=thresh, pthresh=pthresh, pmax=pmax, cbar=cbar, cbar_ax=cbar_ax, 
        cbar_kws=cbar_kws, palette=palette, hue_order=hue_order, 
        hue_norm=hue_norm, color=color, log_scale=log_scale, legend=legend, 
        ax=ax, alpha=alpha, **kwargs
    )    
    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(title, fontweight='bold')

    # Add hue labels
    if hue_labels is not None:
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(
            handles, hue_labels, title=hue, title_fontsize='11', 
            loc='upper right'
        )
        # legend_labels = dict(zip(hue_order, hue_labels))
        # handles, _ = ax.get_legend_handles_labels()
        # labels = [legend_labels.get(int(item), item) for item in handles]
        # ax.legend(handles, labels, title=hue, title_fontsize='11', 
        # loc='upper right')
    
    if add_bar_labels:
        ax = add_barplot_labels(ax, bar_orient, 10)
        
    return ax

# Make a Barplot
def make_barplot(
        data=None,  x=None, y=None, hue=None, order=None, hue_order=None,
        estimator='mean', errorbar=('ci', 95), n_boot=1000, units=None, 
        seed=None, orient=None, color=None, palette=None, saturation=0.75, 
        fill=True, hue_norm=None, width=0.8, dodge='auto', gap=0, 
        log_scale=None, native_scale=False, formatter=None, legend='auto', 
        capsize=0, err_kws=None, ax=None, title=None, xlabel=None, ylabel=None, 
        add_bar_labels=False, xlabel_rot=None
    ):    
    print("="*60)
    print(f"Making a Barplot of '{x}' vs '{y}' with '{hue}' hue\n")
    sns.barplot(
        data=data, x=x, y=y, hue=hue, order=order, hue_order=hue_order, 
        estimator=estimator, errorbar=errorbar, n_boot=n_boot, units=units, 
        seed=seed, orient=orient, color=color, palette=palette, 
        saturation=saturation, fill=fill, hue_norm=hue_norm, width=width, 
        dodge=dodge, gap=gap, log_scale=log_scale, native_scale=native_scale, 
        formatter=formatter, legend=legend, capsize=capsize, err_kws=err_kws, 
        ax=ax
    )

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontweight='bold')

    if ylabel is not None:
        ax.set_ylabel(ylabel, fontweight='bold')

    if title is not None:
        ax.set_title(title, fontweight='bold')

    if xlabel_rot is not None:
        tick_pos = ax.get_xticks()
        tick_lab = ax.get_xticklabels()

        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lab, rotation=xlabel_rot, horizontalalignment='center')
        
    # if hue_labels:
    #     handles, _ = ax.get_legend_handles_labels()
    #     ax.legend(handles, hue_labels, title=hue, title_fontsize='11', loc='best')
    #     # legend_labels = dict(zip(hue_order, hue_labels))
    #     # handles, _ = ax.get_legend_handles_labels()
    #     # labels = [legend_labels.get(int(item), item) for item in handles]
    #     # ax.legend(handles, labels, title=hue, title_fontsize='11', loc='upper right')

    if add_bar_labels:
        ax = add_barplot_labels(ax, orient, 10)
        
    return ax

# Make a Countplot
def make_countplot(
        data=None, x=None, y=None, hue=None, order=None, hue_order=None, 
        orient='v', color=None, palette=None, saturation=0.75, fill=True, 
        hue_norm=None, stat='count', width=0.8, dodge='auto', gap=0, 
        log_scale=None, native_scale=False, formatter=None, legend='auto',
        xlabel=None, ylabel=None, title=None, ax=None, add_bar_labels=False, 
        **kwargs):
    print("="*60)
    print(f"Making a Histogram of '{x}' with '{hue}' hue\n")   
    sns.countplot(
        data=data, x=x, y=y, hue=hue, order=order, orient=orient, color=color,
        palette=palette, saturation=saturation, fill=fill, hue_norm=hue_norm, 
        stat=stat, width=width, dodge=dodge, gap=gap, log_scale=log_scale, 
        native_scale=native_scale, formatter=formatter, legend=legend, ax=ax, 
        **kwargs
    )
    
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontweight='bold')

    if ylabel is not None:
        ax.set_ylabel(ylabel, fontweight='bold')

    if title is not None:
        ax.set_title(title, fontweight='bold')

    # if hue_labels:
    #     handles, _ = ax.get_legend_handles_labels()
    #     ax.legend(handles, hue_labels, title=hue, title_fontsize='11', loc='upper right')
    #     # legend_labels = dict(zip(hue_order, hue_labels))
    #     # handles, _ = ax.get_legend_handles_labels()
    #     # labels = [legend_labels.get(int(item), item) for item in handles]
    #     # ax.legend(handles, labels, title=hue, title_fontsize='11', loc='upper right')

    if add_bar_labels:
        ax = add_barplot_labels(ax, orient, 10)
        
    return ax

# Make a Boxplot
def make_boxplot(
        df, x, y, xtick_vals, xtick_labels, xlabel, ylabel, title, hue=None, 
        order=None, hue_order=None, orient=None, color=None, palette=None, 
        saturation=0.75, fill=True, dodge='auto', width=0.8, gap=0, whis=1.5, 
        linecolor='auto', linewidth=None, fliersize=None, hue_norm=None,
        native_scale=False, log_scale=None, formatter=None, legend='auto', 
        ax=None, **kwargs):    
    print("="*60)
    print(f"Making a Box Plot of '{y}' vs '{x}'\n")        
    sns.boxplot(
        data=df, x=x, y=y, hue=hue, order=order, hue_order=hue_order, 
        orient=orient, color=color,
        palette=palette, saturation=saturation, fill=fill, dodge=dodge, 
        width=width, gap=gap, whis=whis, linecolor=linecolor, 
        linewidth=linewidth, fliersize=fliersize, hue_norm=hue_norm,
        native_scale=native_scale, log_scale=log_scale, formatter=formatter,
        legend=legend, ax=ax, **kwargs
    )
    
    if xtick_vals is not None:
        ax.set_xticks(xtick_vals, xtick_labels)

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontweight='bold')

    if ylabel is not None:
        ax.set_ylabel(ylabel, fontweight='bold')

    if title is not None:
        ax.set_title(title, fontweight='bold')
    
    return ax

# Make a pie chart
def make_piechart(
        x, fig, ax=None, data=None, explode=None, labels=None, colors=None, 
        autopct=None, pctdistance=0.6, shadow=False, labeldistance=1.1, 
        startangle=0, radius=1, counterclock=True, wedgeprops=None, 
        textprops=None, center=(0, 0), frame=False, rotatelabels=False, *,
        normalize=True, hatch=None, title=None):
    plt.pie(
        data=data, x=x, explode=explode, labels=labels, colors=colors, 
        autopct=autopct,pctdistance=pctdistance, shadow=shadow, 
        labeldistance=labeldistance, startangle=startangle,
        radius=radius, counterclock=counterclock, wedgeprops=wedgeprops, 
        textprops=textprops, center=center, frame=frame, 
        rotatelabels=rotatelabels, normalize=normalize, hatch=hatch
    )

    if title is not None:
        plt.title(title, fontweight='bold')

    return fig, ax
        
