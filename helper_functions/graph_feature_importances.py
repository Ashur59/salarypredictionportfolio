# +
import pandas as pd

def graph_feature_importances(model, feature_names, autoscale=True, headroom=0.1, width=10, summarized_columns=None):
    feature_dict=dict(zip(feature_names, model.feature_importances_))
    
    if summarized_columns:
        for col_name in summarized_columns:
            sum_value = sum(x for i, x in feature_dict.items() if col_name in i )
            keys_to_remove = [i for i in feature_dict.keys() if col_name in i ]
            for i in keys_to_remove:
                feature_dict.pop(i)
            feature_dict[col_name] = sum_value
    results = pd.Series(feature_dict, index=feature_dict.keys())
    results.sort_values(inplace=True)
    #print(results)
    n_features = len(feature_names)-1
    ax = results.plot(kind='barh', figsize=(width, len(results)/2), 
                       xlim=(0, .30), 
                       ylim=[-1, n_features], 
                       label=model.__class__.__name__)
    ax.set_xlabel("Feature importance")
    ax.set_ylabel("Feature")
    ax.get_figure().savefig("images/Feature_Importances.png", dpi=None, facecolor='w', edgecolor='w', 
            orientation='portrait', papertype=None, format=None,
            transparent=False, bbox_inches='tight', pad_inches=0,
            frameon=None, metadata=None)
