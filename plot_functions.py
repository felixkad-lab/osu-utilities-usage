import numpy as np
import matplotlib.pyplot as plt
import graphing_helper as gr
from ml_helper import do_linear_regression, regression_by_parts

# Graph of Credit (GHS) vs day (Raw and Fixed)
def plot_raw_and_fixed_credits(
          df=None, x_lim=None, y_lim=None, xtick_step=None, figsize=None,figname=None):
    print("="*60)
    print("Making a graph of Credit vs Day for Raw and Fixed Data\n")
        
    fig, axes = plt.subplots(2, 1, figsize=figsize)
        
    x_min = x_lim[0]; x_max = x_lim[1]
    y_min = y_lim[0]; y_max = y_lim[1]
    xlabels = ['', 'Day Count']
    ylabels = ['Credit (GHS)', 'Credit (GHS)']
    titles = [
        'Credit (GHS) vs Day for Raw Data', 
        'Credit vs Day (Normalized baseline)'
    ]
    xlim = [x_min, x_max]
    ylim = [y_min, y_max]
    xtick_values = np.arange(0, x_max + 1, 20)
    xtick_labels = [str(val) for val in xtick_values]
    
    for i, col in enumerate(['credit_ghs', 'credit_ghs_fixed']):
        ax = axes[i]
        ax.scatter(df['day_count'], df[col], color='tab:blue')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(xlabels[i], fontweight='bold', fontsize=14)
        ax.set_ylabel(ylabels[i], fontweight='bold', fontsize=14)
        ax.set_title(titles[i], fontweight='bold', fontsize=18)
            
        # xtick marks
        ax.set_xticks(xtick_values)
        ax.set_xticklabels(xtick_labels)        
            
    # Save figure    
    plt.tight_layout(h_pad=2)
    plt.savefig(figname, dpi=300, bbox_inches='tight')
    plt.close(fig)

# Make a Countplot
def make_countplot(
        data=None, x=None, y=None, hue=None, order=None,hue_order=None,
        orient='v', color=None, palette=None, saturation=0.75, fill=True, hue_norm=None, stat='count', width=0.8, dodge='auto',
        gap=0, log_scale=None, native_scale=False, formatter=None, legend='auto', figsize=[8, 6], figname=None, add_bar_labels=True, xlabel=None, ylabel=None, title=None):
    fig, ax = plt.subplots(figsize=figsize)
    ax = gr.make_countplot(
        data=data, x=x, y=y, hue=hue, order=order, hue_order=hue_order, orient=orient, color=color, palette=palette, saturation=saturation, fill=fill, hue_norm=hue_norm,
        stat=stat, width=width, dodge=dodge, gap=gap,log_scale=log_scale, native_scale=native_scale, formatter=formatter, legend=legend,
        add_bar_labels=add_bar_labels, xlabel=xlabel, ylabel=ylabel, title=title, ax=ax
    )

    # Save graph    
    plt.savefig(figname, dpi=300, bbox_inches='tight')
    plt.close(fig)

# Make a histogram of Time elapsed
def time_elapsed_histogram(
        df=None, bins='auto', binrange=None, figsize=None, figname=None):
    print("="*60)
    print("Making a Histogram of time elapsed\n")

    fig, ax = plt.subplots(figsize=figsize)
    ax = gr.make_histogram(
        df=df, x='time_elapsed', stat='percent', bins=bins, binrange=binrange,
        xlabel='Time Elapsed (hr)', ylabel='Frequency (%)', 
        add_bar_labels=False, title='Histogram of Time Elapsed', ax=ax
    )
    
    # Save graph
    plt.savefig(figname, dpi=300, bbox_inches='tight')
    plt.close(fig)

# Make pie chart of Time elapsed
def plot_time_elapsed_pie(df=None, bin_edges=None, figsize=None, figname=None):
    print("="*60)
    print("Making a Pie Chart of time elapsed\n")
    
    # Create labels based on bin_edges
    bin_labels = []
    for i in range(len(bin_edges) - 1):
        label = str(bin_edges[i]) + " - " + str(bin_edges[i+1])
        bin_labels.append(label)
    
    # Pie chart
    fig, ax = plt.subplots(figsize=figsize)
    fig, ax = gr.make_piechart(
        x='perc', fig=fig, ax=ax, data=df, labels=df['labels'],
        title='Time Elapsed categories (hours)', autopct='%1.1f%%', startangle=140, shadow=False
    )
        
    # Save graph
    plt.savefig(figname, dpi=300, bbox_inches='tight')
    plt.close(fig)

# Make a Histogram
def make_histogram(
        df=None, x=None, figsize=None, figname=None, hue_order=None, hue_labels=None, add_bar_labels=None, hue=None, stat='count', palette=None, alpha=1, binrange=None, bins='auto', xlabel=None, ylabel=None, title=None, discrete=True):
        # Create figure and axes
        fig, ax = plt.subplots(figsize=figsize)
        ax = gr.make_histogram(
            df=df, x=x, hue=hue, stat=stat, palette=palette, alpha=alpha,
            binrange=binrange, bins=bins, xlabel=xlabel, ylabel=ylabel, title=title, hue_order=hue_order, hue_labels=hue_labels, 
            discrete=discrete, add_bar_labels=add_bar_labels, ax=ax
        )
        
        # Save graph    
        plt.savefig(figname, dpi=300, bbox_inches='tight')
        plt.close(fig)

# Make a Boxplot
def make_boxplot(
        df=None, x=None, y=None, xtick_vals=None, xtick_labels=None,
        xlabel=None, ylabel=None, title=None, figsize=None, figname=None):
    fig, ax = plt.subplots(figsize=figsize)
    ax = gr.make_boxplot(
        df=df, x=x, y=y, xtick_vals=xtick_vals, xtick_labels=xtick_labels,
        xlabel=xlabel, ylabel=ylabel, title=title, ax=ax
    )
        
    # Save graph
    plt.savefig(figname, dpi=300, bbox_inches='tight')
    plt.close(fig)

# Plot Weekly usage
def plot_weekly_usage(df=None, figsize=None, figname=None):
    print("="*60)
    print("Making a plot of Weekly used\n")
    # Make graph
    fig, ax = plt.subplots(figsize=figsize)
    ax = gr.make_barplot(
        data=df, x='week', y='ghs_used', orient='v', xlabel='Week',
        ylabel='Credit Used (GHS)', title='Weekly Usage (GHS)', ax=ax, add_bar_labels=False
    )
        
    # Save graph
    plt.savefig(figname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    

# Make a graph of date vs ghs_used_monthly
def plot_monthly_usage(df=None, figsize=None, figname=None):
    df['date'] = df['date'].dt.strftime('%b %y')
        
    fig, ax = plt.subplots(figsize=figsize)
    ax = gr.make_barplot(
        data=df, x='date', y='ghs_used', orient='v', xlabel='Month',
        ylabel='Credit Used (GHS)', title='Monthly Usage (GHS)', ax=ax, add_bar_labels=False, xlabel_rot=45
    )
        
    # save graph
    plt.savefig(figname, dpi = 300, bbox_inches='tight')
    plt.close(fig)

# Make a scatter plot with linear regression results
def linreg_and_graph(
        df=None, xcol=None, ycol=None, pred_y=None, m=None, b=None, rmse=None,
        r2=None, xlabel=None, ylabel=None, title=None, eqn_info=None, 
        eqn_loc=None, r2_loc=None, mse_loc=None, 
        rmse_loc=None, scat_color=None, line_color=None, x_lim=None, y_lim=None,
        xtick_step=None, figsize=None, figname=None): 
    print("="*60)
    print(f"Linear Regression for {ycol} vs {xcol}\n")
    pred_y, m, b, rmse, r2, scipy_linreg = do_linear_regression(
        df[xcol], df[ycol]
    )

    # Convert linear regression results to list for graphing
    x_list = [df[xcol]]
    y_list = [df[ycol]]
    y_pred = [pred_y]
    m_list = [m]
    b_list = [b]
    scipy_linreg = [scipy_linreg]
    
    # Create a graph
    print("Making a graph for the regression\n")
    y_symb = eqn_info[0]; x_symb = eqn_info[1]; m_units = eqn_info[2]; 
    b_units = eqn_info[3]

    # Create fig canvas and axes
    fig, ax = plt.subplots(figsize=figsize)

    lr_eqn = [
        fr"$ {y_symb} = ({m:.4g} \, {m_units}) \cdot {x_symb} + "
        fr"({b:.2f} \, {b_units}) $"
        for _ in range(len(x_list))
    ]
    r2_eqn = [fr'$ R^2 = {r2:.4g} $' for _ in range(len(x_list))]
    rmse_eqn = [fr'$ RMSE = {rmse:.4g} $' for _ in range(len(x_list))]
        
    # Make a scatter plot with line of best fit
    fig, ax = gr.make_scatter_plot(
        fig, ax, x=x_list, y=y_list, y_pred=y_pred, scat_color=scat_color, 
        line_color=line_color, xlabel=xlabel, ylabel=ylabel, title=title, 
        x_lim=x_lim, y_lim=y_lim, add_eqns=True, eqn_loc=eqn_loc, 
        r2_loc=r2_loc, xticks=np.arange(x_lim[0], x_lim[1] + 1, xtick_step),
        rmse_loc=rmse_loc, r2_eqn=r2_eqn, rmse_eqn=rmse_eqn, lr_eqn=lr_eqn
    )
        
    # Save graph
    plt.savefig(figname, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return x_list, y_list, y_pred, m_list, b_list, scipy_linreg    

# Linear Regression of Credit vs days for Electricity by Parts and Graph
def linreg_and_graph_parts(
        df=None, xcol=None, ycol=None, part_col=None, xlabel=None, 
        ylabel=None, title=None, eqn_info=None, eqn_loc=None, r2_loc=None, 
        mse_loc=None, rmse_loc=None, part_loc=None, scat_color=None, 
        line_color=None, x_lim=None, y_lim=None, 
        xtick_step=None, figsize=None, figname=None):
    print("="*60)
    print(f"Linear Regression for {ycol} vs {xcol} by parts\n")

    # Count unique number of parts
    num_unique_parts = df[part_col].nunique()
    unique_values = df[part_col].unique().tolist()

    parts = [None] * num_unique_parts
    part_init_xval = [None] * (num_unique_parts)
    for i, value in enumerate(unique_values):
        df_filtered = df[df[part_col] == value]
        parts[i] = df_filtered
        part_init_xval[i] = parts[i].loc[parts[i].index[0], xcol]
    
    # Do Linear regression for each part
    x_list, y_list, y_pred, m_list, b_list, rmse_list, r2_list, scipy_linreg =\
        regression_by_parts(parts, xcol, ycol, part_col)
    
    # Equation information
    print("Making a graph for the regression\n")
    y_symb = eqn_info[0]
    x_symb = eqn_info[1]
    m_units = eqn_info[2]
    b_units = eqn_info[3]
    lr_eqn = [
        fr"$ {y_symb} = ({m_list[i]:.3g} \, {m_units}) \cdot {x_symb}"
        fr" + ({b_list[i]:.4g} \, {b_units}) $"
        for i in range(len(x_list))
    ]
    r2_eqn = [fr'$ R^2 = {r2_list[i]:.4g} $' for i in range(len(x_list))]
    rmse_eqn = [
        fr'$ RMSE = {rmse_list[i]:.4g} $' for i in range(len(x_list))
    ]
        
    # Create fig canvas and axes
    fig, ax = plt.subplots(figsize=figsize)
    xticks = np.arange(x_lim[0], x_lim[1] + 1, xtick_step)
    fig, ax = gr.make_scatter_plot(
        fig, ax, x=x_list, y=y_list, y_pred=y_pred, scat_color=scat_color, 
        line_color=line_color, xlabel=xlabel, ylabel=ylabel, title=title, 
        x_lim=x_lim, y_lim=y_lim, xticks=xticks, add_eqns=True, 
        eqn_loc=eqn_loc, r2_loc=r2_loc, rmse_loc=rmse_loc, lr_eqn=lr_eqn,
        r2_eqn=r2_eqn, rmse_eqn=rmse_eqn
    )
        
    # Include Vertical line separator
    for i in range(1, num_unique_parts, 1):
        ax.axvline(
            x=part_init_xval[i], color='lightgrey', linestyle='--', label=''
        )

    # Add text for Parts
    for i in range(len(x_list)):
        ax.text(
            part_loc[i][0], part_loc[i][1], f'Part {i+1}', color=scat_color[i], 
            fontsize=14, fontname='serif', fontweight='bold'
        )
        
    # Save graph
    plt.savefig(figname, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return x_list, y_list, y_pred, m_list, b_list, scipy_linreg
    