from MDA import MultilinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

class MDATrainer:
    def __init__(self, classifiers, ranks, total_iters=5):
        self.classifiers = classifiers
        self.ranks = ranks
        self.total_iters = total_iters

    # Cross-validation to find optimal hyperparameters
    def apply_mda_and_scale(self, train_data, train_lbls, test_data, test_lbls, rank):
        mda = MultilinearDiscriminantAnalysis(total_iters=self.total_iters, rank=rank)
        mda.fit(train_data, train_lbls)
            
        # Transform train and val
        X_train_trans = mda.transform(train_data)
        X_val_trans = mda.transform(test_data)

        # Scale (fit on train only)
        scaler_temp = StandardScaler()
        scaler_temp.fit(X_train_trans)
        X_train_scaled = scaler_temp.transform(X_train_trans)
        X_val_scaled = scaler_temp.transform(X_val_trans)

        return X_train_scaled, X_val_scaled, mda
    
    def tune_and_classify(self, X_train, y_train, X_val, y_val):
        best_clf = None
        best_clf_name = None
        best_params = None
        best_score = 0
        
        for clf_name, (clf, param_grid) in self.classifiers.items():
            # Only use GridSearch if there are parameters to tune
            if param_grid:
                grid_search = GridSearchCV(clf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
                grid_search.fit(X_train, y_train)
                score = grid_search.best_score_
                if score > best_score:
                    best_score = score
                    best_clf = grid_search.best_estimator_
                    best_params = grid_search.best_params_
                    best_clf_name = clf_name
            else:
                # For classifiers with no hyperparameters
                clf.fit(X_train, y_train)
                score = clf.score(X_val, y_val)
                if score > best_score:
                    best_score = score
                    best_clf = clf
                    best_params = {}
                    best_clf_name = clf_name
        
        print(f"Classifier: {best_clf}, Score: {best_score:.4f}")
        return best_clf, best_score
    
    def run(self, cv_fold):
        best_rank = None
        best_score = 0
        best_clf = None
        print("Rank and Classifier selection:")
        for rank in self.ranks:
            print("Trying rank:", rank)
            X_train_scaled, X_val_scaled, model = self.apply_mda_and_scale(cv_fold.train_data, cv_fold.train_lbls, cv_fold.val_data, cv_fold.val_lbls, rank)
            best_clf, best_rank_score = self.tune_and_classify(X_train_scaled, cv_fold.train_lbls, X_val_scaled, cv_fold.val_lbls)

            if best_rank_score > best_score:
                best_score = best_rank_score
                best_rank = rank
                best_clf = best_clf
        return best_rank, best_clf, best_score