

def list_outliers(col, results, z_threshold=3):
    df = results.pivot_table(index=['type','task', 'framework'], columns='fold', values=col)
    df_mean = df.mean(axis=1)
    df_std = df.std(axis=1)
    z_score = (df.sub(df_mean, axis=0)
               .div(df_std, axis=0)
               .abs())
    return z_score.where(z_score > z_threshold).dropna(axis=0, how='all')
