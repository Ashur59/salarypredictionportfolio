from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV, ShuffleSplit, RandomizedSearchCV

seed = 7
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
def randomized_search(model, distribution, X_train, y_train, X_validation, y_validation):
    randomized_search = RandomizedSearchCV(model, distribution, cv=kfold, 
                                           return_train_score=True, n_jobs=-1, scoring='neg_mean_squared_error')
    search = randomized_search.fit(X_train, y_train)
    
    print("Best estimator:\n{} \
           \nBest parameters:\n{} \
           \nBest cross-validation score: {:.3f} \
           \nBest test score: {:.3f}\n\n".format(search.best_estimator_, 
                                                 search.best_params_, 
                                                 -1*search.best_score_, 
                                                 -1*search.score(X_validation, y_validation)
                                                )
         )
