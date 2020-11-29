cols = df_train2.columns
cols = cols.drop(["Rings", "sqrt_log_Rings"])

n=4

col_names = ["R²"] + [f"Variable {i}" for i in range(1, n+1)]
print(col_names)

def name_array_gen(n, col_names):
    cols_dict = {}
    for k in range(1, n+1):
        cols_dict[k] = col_names
    perm = np.array(np.meshgrid(*(v for _, v in sorted(cols_dict.items())))).T.reshape(-1, n).tolist()
    return perm
    
    
#names = list(map(lambda x: f'sqrt_log_Rings ~ Q("{x[0]}")+Q("{x[1]}")', perm))
def model_applier(x):
    model_name = f'sqrt_log_Rings ~ Q("{x[0]}")'
    for k in range(1, len(x)):
        model_name += f' +Q("{x[k]}")'
    #print(str(x))
    results = smf.ols(model_name, data=df_train2).fit()
    r2 = round(results.rsquared, 3)
    #print(r2)
    return [r2 str(x)]

#perm = name_array_gen(n, cols)
#data = [model_applier(x) for x in perm] 

#res = model_applier(perm
#res = np.apply_along_axis(model_applier_2_var, 1, names)

#res_df = pd.DataFrame(data=data, columns=col_names)
#res_df = res_df.sort_values(by=["R²"], ascending=False)
#print(res_df.head(20))

def best_models(cols, n_max=3):
    #col_names = ["R²"] + [f"Variable {i}" for i in range(1, n_max+1)]
    #final_df = pd.DataFrame(columns=col_names)
    for k in range(2, n_max+1):
        col_names = ["R²"] + [f"Variable {i}" for i in range(1, k+1)]
        perm = name_array_gen(k, cols)
        data = [model_applier(x) for x in perm]
        temp_df = pd.DataFrame(data=data, columns=col_names)
        temp_df = temp_df[temp_df["R²"]==max(temp_df["R²"])]
        temp_df = temp_df.sort_values(by=["R²"], ascending=False)
        print(temp_df)
        #temp_df.loc[0, col_names[:k+1]] = data
        #final_df.concat(temp_df)

    #return final_df

def best_models_2(cols, n_max=3):
    col_names = ["R²", "Model"]
    #col_names = ["R²"] + [f"Variable {i}" for i in range(1, n_max+1)]
    final_df = pd.DataFrame(columns=col_names)
    for k in range(2, n_max+1):
        
        perm = name_array_gen(k, cols)
        data = [model_applier(x) for x in perm]
        temp_df = pd.DataFrame(data=data, columns=col_names)
        #temp_df = temp_df[temp_df["R²"]==max(temp_df["R²"])]
        #temp_df = temp_df.sort_values(by=["R²"], ascending=False)
        final_df = final_df.append(temp_df)
        
    return final_df.sort_values(by=["R²"], ascending=False)
#def give_me_grid(d):
#    return np.array(np.meshgrid(*(v for _, v in sorted(d.items()))))

#print(give_me_grid(dict_1))     


final_df = best_models_2(cols=cols, n_max=3) 
print(final_df.shape)
final_df.head(50),
