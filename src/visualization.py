import seaborn as sns


def plot_variables(df, arg1, arg2=None):
    if ('array' in str(type(df[arg1].values))) and arg2 is None:
        if 'array' in str(type(df[arg1].values)):
            sns.distplot(df[arg1].dropna())
        else:
            sns.barplot(df[arg1].dropna())
    elif ('array' in str(type(df[arg1].values))) and ('array' in str(type(df[arg2].values))):
        sns.jointplot(arg1, arg2, df)
    elif ('array' in str(type(df[arg1].values))) or ('array' in str(type(df[arg2].values))):
        if 'array' in str(type(df[arg2].values)):
            arg_cat = arg1
            arg_num = arg2
        else:
            arg_cat = arg2
            arg_num = arg1
        sns.boxplot(arg_cat, arg_num, data=df)
    else:
        print("No plot possible, at least one argument must be numerical")
