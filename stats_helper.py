from scipy import stats

# Perform T-Test
def do_t_test(
        data1, data2, data1_name, data2_name, axis=0, equal_var=True, nan_policy='propagate', permutations=None, random_state=None, alternative='two-sided', trim=0, keepdims=False, alpha=None,outname=None):
    print(
        f"Performing T-Test analysis for '{data1_name}' and '{data2_name}'\n"
    )
    
    # Do test
    t_stat, p_value = stats.ttest_ind(
        data1, data2, axis=axis, equal_var=equal_var, nan_policy=nan_policy ,
        permutations=permutations, random_state=random_state,
        alternative=alternative, trim=trim, keepdims=False
    )

    # Interpret results
    if outname is not None:
        with open(outname, "w") as file:
            file.write("="*60)
            file.write("\n\tT-Test Results\n")
            file.write("="*60)
            file.write(
                f"\nMean of {data1_name} = {data1['ghs_used'].mean()}\n"
            )
            file.write(f"Mean of {data2_name} = {data2['ghs_used'].mean()}\n")
            file.write(
                f"\nVariance of {data1_name} = {data1['ghs_used'].var()}\n"
            )
            file.write(
                f"Variance of {data2_name} = {data2['ghs_used'].var()}\n"
            )
            file.write(f"\nT-statistics: {t_stat[0]}\n")
            file.write(f"P-value: {p_value[0]}\n")
            if p_value[0] < alpha:
                file.write(
                    f"\nReject the null hypothesis, i.e., there is a "
                    f"significant difference between {data1_name} and "
                    f"{data2_name}\n"
                )
            else:
                file.write(
                    f"\nFail to reject the null hypothesis, i.e.. there is no "
                    f"significant difference between {data1_name} and "
                    f"{data2_name}\n"
                )
        print(f"T-Test results saved to {outname}\n")
    else:
        print("="*60)
        print("\n\tT-Test Results\n")
        print("="*60)
        print(f"\nMean of {data1_name} = {data1['ghs_used'].mean()}")
        print(f"\nMean of {data2_name} = {data2['ghs_used'].mean()}")
        print(f"\nVariance of {data1_name} = {data1['ghs_used'].var()}")
        print(f"\nVariance of {data2_name} = {data2['ghs_used'].var()}")
        print(f"\nT-statistics: {t_stat[0]}\n")
        print(f"P-value: {p_value[0]}\n")
        if p_value[0] < alpha:
            print(
                f"\nReject the null hypothesis, i.e., there is a significant "
                f"difference between {data1_name} and {data2_name}\n"
            )
        else:
            print(
                f"\nFail to reject the null hypothesis, i.e., there is no "
                f"significant difference between {data1_name} and {data2_name}"
                "\n"
            )
                    
    # Return results
    return t_stat, p_value

# Perform Anova-Test
def do_anova_test(data, data_names, axis=0, outname=None):
    #print(f"Performing Anova-Test analysis for '{data1_name}' and '{data2_name}'\n")
    ## Do test
    f_stat, p_value = stats.f_oneway(
        data[0], data[1], data[2], data[3], axis=axis
    )

    # ## Interpret results
    # if outfile:
    #     with open(outfile, "w") as file:
    #         file.write("="*60)
    #         file.write("\n\tT-Test Results\n")
    #         file.write("="*60)
    #         file.write(f"\nMean of {data1_name} = {data1['ghs_used'].mean()}\n")
    #         file.write(f"\Mean of {data2_name} = {data2['ghs_used'].mean()}\n")
    #         file.write(f"\nVariance of {data1_name} = {data1['ghs_used'].var()}\n")
    #         file.write(f"\Variance of {data2_name} = {data2['ghs_used'].var()}\n")
    #         file.write(f"\nT-statistics: {t_stat[0]}\n")
    #         file.write(f"P-value: {p_value[0]}\n")
    #         if p_value[0] < alpha:
    #             file.write(f"\nReject the null hypothesis, i.e., there is a significant difference between {data1_name} and {data2_name}\n")
    #         else:
    #             file.write(f"\nFail to reject the null hypothesis, i.e.. there is no significant difference between {data1_name} and {data2_name}\n")
    #     print(f"T-Test results saved to {outfile}\n")
    # else:
    #     print("="*60)
    #     print("\n\tT-Test Results\n")
    #     print("="*60)
    #     print(f"\nMean of {data1_name} = {data1['ghs_used'].mean()}")
    #     print(f"\nMean of {data2_name} = {data2['ghs_used'].mean()}")
    #     print(f"\nVariance of {data1_name} = {data1['ghs_used'].var()}")
    #     print(f"\nVariance of {data2_name} = {data2['ghs_used'].var()}")
    #     print(f"\nT-statistics: {t_stat[0]}\n")
    #     print(f"P-value: {p_value[0]}\n")
    #     if p_value[0] < alpha:
    #         print(f"\nReject the null hypothesis, i.e., there is a significant difference between ",
    #               f"{data1_name} and {data2_name}\n")
    #     else:
    #         print(f"\nFail to reject the null hypothesis, i.e.. there is no significant difference between ",
    #               f"{data1_name} and {data2_name}\n")
                    
    ## Return results
    return f_stat, p_value