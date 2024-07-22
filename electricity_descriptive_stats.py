import math
import helper_functions as helper
import plot_functions as pf

def do_elec_descriptive_stats(prefix=None, df_util=None, top_up=None):
    # Make a Graph of credit vs day (raw data and normalized baseline)    
    fig_ctr = 0
    x_lim = [0, 420]
    y_lim = [0, 8600]
    figname = (
        f'output/{prefix}_{str(fig_ctr).zfill(2)}_raw_and_fixed.png'
    )       
    pf.plot_raw_and_fixed_credits(
        df=df_util, x_lim=x_lim, y_lim=y_lim, xtick_step=20, figsize=[10, 12], figname=figname
    )
        
    # Make a countplot of measured and calculated values   
    fig_ctr = 1
    title = 'Counts of Measured values (=1) and Calculated values (=0)'
    figname = (
        f'output/{prefix}_{str(fig_ctr).zfill(2)}_measured_countplot.png'
    )
    pf.make_countplot(
        data=df_util, x='measured', hue=None, stat='count', palette=None, saturation=1, title=title, xlabel='Measurement', ylabel='Count', figsize=[8, 6], add_bar_labels=True, hue_order=None, figname=figname
    )
      
    # Make a histogram of AC usage  
    fig_ctr = 2
    title = 'Counts of Measurements with AC use (=1) vs No AC use (=0)'
    figname = (
        f'output/{prefix}_{str(fig_ctr).zfill(2)}_ACuse_countplot.png'
    )
    pf.make_countplot(
        data=df_util, x='ac_use', hue=None, stat='count', saturation=1,palette=None, title=title, xlabel='AC use', ylabel='Count', hue_order=None, add_bar_labels=True, figsize=[8, 6], figname=figname
    )

    # Make Histogram of AC usage Disaggregated by measured
    fig_ctr = 3
    title = 'Counts of Measurements with AC (=1) vs No AC use (=0)'
    figname = (
        f'output/{prefix}_{str(fig_ctr).zfill(2)}'
        f'_ACuse_and_measured_countplot.png'
    )
    pf.make_countplot(
        data=df_util, x='ac_use', hue='measured', stat='count', palette=None, saturation=1, xlabel='AC use', ylabel='Count', add_bar_labels=True, hue_order=None, title=title, figsize=[8,6],
        figname=figname
    )      
        
    # Make a histogram of time_elapsed 
    fig_ctr = 4
    figname = (
        f'output/{prefix}_{str(fig_ctr).zfill(2)}_timeElapsed_histogram.png'
    )
    pf.time_elapsed_histogram(
        df=df_util, bins=26, binrange=[10, 36], figsize=[10, 6],
        figname=figname
    )
        
    # Calculate time elapsed percentages for pie chart
    min_et = df_util['time_elapsed'].min(); min_et=math.floor(min_et)
    max_et = df_util['time_elapsed'].max(); max_et=math.ceil(max_et)
    bin_edges=[min_et, 18, 22, 23, 25, 26, max_et]
    outname = f'output/{prefix}_timeElapsed.tsv'
    elapsed_time_df = helper.calc_time_elapsed_pie(
        df=df_util, bin_edges=bin_edges, outname=outname
    )

    # Make a pie chart of time_elapsed
    fig_ctr = 5
    figname = (
        f'output/{prefix}_{str(fig_ctr).zfill(2)}_timeElapsed_piechart.png'
    )
    pf.plot_time_elapsed_pie(
        df=elapsed_time_df, bin_edges=bin_edges, figname=figname, figsize=[6, 6]
    )
        
    # Calculate weekly usage
    outname = f'output/{prefix}_weekly.tsv'
    weekly_usage_df = helper.calc_weekly_usage(df=df_util, outname=outname)
    
    # Plot Credit Usage by week  
    fig_ctr = 6
    figname = (
        f'output/{prefix}_{str(fig_ctr).zfill(2)}_weekly_usage.png'
    )
    pf.plot_weekly_usage(
        df=weekly_usage_df, figsize=[20, 10], figname=figname
    )

    # Calculate monthly credit usage
    outname = f'output/{prefix}_monthly.tsv'
    monthly_usage_df = helper.calc_monthly_usage(
        prefix, df=df_util, top_up=top_up, outname=outname
    )

    # Plot credit usage by month
    fig_ctr = 7
    figname=f'output/{prefix}_{str(fig_ctr).zfill(2)}_monthly_usage.png'
    pf.plot_monthly_usage(
        df=monthly_usage_df, figsize=[8, 6], figname=figname,
    )
        
    # Restrict data to only measured values
    df_util = helper.only_measured(prefix, df=df_util)    
            
    # Perform T-Test for ghs_used for AC vs no AC
    outname = f"output/{prefix}_test_t_AC.txt"
    helper.ac_t_test(prefix, df=df_util, outname=outname)
        
    # Perform anova-Test for ghs_used based on part
    outname = f"output/{prefix}_test_anova_part.txt"
    helper.ac_anova_test(prefix, df=df_util, outname=outname)

    # Histogram of ghs_used 
    fig_ctr = 8
    figname = (
        f'output/{prefix}_{str(fig_ctr).zfill(2)}_creditUsed_histogram.png'
    )
    pf.make_histogram(
        df=df_util, x='ghs_used', hue=None, stat='percent', palette=None, alpha=1, xlabel='Credit Used (GHS)', ylabel='Freqency (%)', 
        binrange=[0,60], bins=30, title='Histogram of Credit Used', hue_order=None, hue_labels=None, figsize=[8, 6], add_bar_labels=False, figname=figname
    )
        
    # Boxplot of ghs_used   
    fig_ctr = 9
    figname = f'output/{prefix}_{str(fig_ctr).zfill(2)}_creditUsed_boxplot.png'
    pf.make_boxplot(
        df=df_util, x=None, y='ghs_used', xtick_vals=None, xtick_labels=None, xlabel=None, ylabel='Credit Used (GHS)', figsize=[4,6],
        title='Box Plot of Credit Used', figname=figname
    )
        
    # Histogram of ghs_used for AC and no AC  
    fig_ctr = 10
    figname = (
        f'output/{prefix}_{str(fig_ctr).zfill(2)}'
        '_creditUsed_byAC_histogram.png'
    )
    pf.make_histogram(
        df=df_util, x='ghs_used', hue='ac_use', stat='percent', palette='deep',
        alpha=1, binrange=[0,60], bins=30, xlabel='Credit Used (GHS)', hue_order=[0, 1], hue_labels=None, figsize=[8, 6], 
        ylabel='Freqency (%)', add_bar_labels=False,
        title='Histogram of Credit Used with and without AC', figname=figname
    )
            
    # Boxplot of ghs_used for AC and no AC
    fig_ctr = 11
    figname = (
        f'output/{prefix}_{str(fig_ctr).zfill(2)}_creditUsed_vs_AC_boxplot.png'
    )
    pf.make_boxplot(
        df=df_util, x='ac_use', y='ghs_used', xtick_vals=[0, 1],
        xlabel='AC Usage', xtick_labels=['Without AC', 'With AC'], ylabel='Credit Used (GHS)', figsize=[8, 6], 
        title='Box Plot of Credit Used with and w/out AC', figname=figname
    )
            
    # Histogram of ghs_used based on part
    fig_ctr = 12
    figname = (
        f'output/{prefix}_{str(fig_ctr).zfill(2)}'
        '_creditUsed_byPart_histogram.png'
    )
    pf.make_histogram(
        df=df_util, x='ghs_used', hue='part', stat='percent', palette='deep',
        alpha=0.5, binrange=[0, 60], bins=30, xlabel='Credit Used (GHS)', figsize=[8, 6], ylabel='Freqency (%)', 
        title='Histogram of Credit Used disaggregated by Part',
        hue_order=[1, 2, 3, 4], hue_labels=None, add_bar_labels=False, figname=figname
    )
            
    # Boxplot of ghs_used for part
    fig_ctr = 13
    xtick_labels = [
        'Flatmate1 - AC', 'w/Flatmate1 - No AC', 'w/Flatmate2 - AC', 
        'Living alone - AC'
    ]
    figname = (
        f'output/{prefix}_{str(fig_ctr).zfill(2)}'
        '_creditUsed_vs_part_boxplot.png'
    )
    pf.make_boxplot(
        df=df_util, x='part', y='ghs_used', xtick_vals=[0, 1, 2, 3],
        xlabel='Data Part', ylabel='Credit Used (GHS)', figsize=[8, 6], 
        xtick_labels=xtick_labels, figname=figname,
        title='Box Plot of Credit Used vs different roommates'
    )
