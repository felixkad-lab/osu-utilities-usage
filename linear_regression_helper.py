from plot_functions import linreg_and_graph, linreg_and_graph_parts

# Linear regression of Credit vs Day_count and graph for all data
def elec_linear_regression_all(df_util=None, figname=None):
    x_list, y_list, y_pred, m_list, b_list, scipy_linreg =\
        linreg_and_graph(
            df=df_util, xcol='day_count', ycol='credit_ghs_fixed', 
            xlabel='Day Count', ylabel='Credit (GHS)', x_lim=[0, 420], 
            y_lim=[0, 8600], xtick_step=20, eqn_loc=[40, 4600], 
            rmse_loc=[4200], r2_loc=[3800], figsize=[16, 10], 
            title='Linear Regression of Credit vs Day', 
            eqn_info=['C', 'd', r"\frac{{GHS}}{{day}}", 'GHS'], 
            line_color=['black'], scat_color=['tab:blue'], figname=figname
        )

# Linear regression of Credit vs Day_count and graph for different flatmate 
# situation
def elec_linear_regression_parts(df_util=None, figname=None):   
    part_loc =[
        [5, 8400], 
        [50, 8400], 
        [170, 8400], 
        [315, 8400]
    ]
    eqn_loc = [
        [5, 6100],
        [50, 8100],
        [170, 8100], 
        [315, 8100]
    ]
    r2_loc = [5700, 7700, 7700, 7700]
    rmse_loc = [5300, 7300, 7300, 7300]
    x_list, y_list, y_pred, m_list, b_list, scipy_linreg =\
    linreg_and_graph_parts(
        df=df_util, xcol='day_count', ycol='credit_ghs_fixed', part_col='part', 
        xlabel='Day Count', ylabel='Credit (GHS)', x_lim=[0, 420], 
        y_lim=[0, 8600], xtick_step=20,  eqn_loc=eqn_loc, r2_loc=r2_loc, 
        rmse_loc=rmse_loc, part_loc=part_loc,  figsize=[16, 10],
        title='Linear Regression of Credit vs Day by Parts', 
        eqn_info=['C', 'd', r"\frac{{GHS}}{{day}}", 'GHS'],
        scat_color=['tab:blue', 'tab:orange', 'tab:green', 'tab:purple'],
        line_color=['black', 'black', 'black', 'black'], figname=figname
    )
    return m_list, b_list, scipy_linreg

# Linear regression of Credit vs Day_count and graph for all data
def wifi_linear_regression_all(df_util=None, figname=None):
    x_list, y_list, y_pred, m_list, b_list, scipy_linreg =\
        linreg_and_graph(
            df=df_util, xcol='day_count', ycol='credit_ghs_fixed', 
            xlabel='Day Count', ylabel='Credit (GB)', x_lim=[0, 360], 
            y_lim=[0, 4000], xtick_step=20, eqn_loc=[40, 1000], 
            rmse_loc=[800], r2_loc=[600], figsize=[16, 10], 
            title='Linear Regression of Credit vs Day', 
            eqn_info=['C', 'd', r"\frac{{GHS}}{{day}}", 'GB'], 
            line_color=['black'], scat_color=['tab:blue'], figname=figname
        )
    return m_list, b_list, scipy_linreg