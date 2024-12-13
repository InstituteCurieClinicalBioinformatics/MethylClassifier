# lib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from argparse import ArgumentParser
import umap
import pickle
import os

class MethylClassifier:
    def __init__(self):        
        self.args = self.get_args()
        self.betaMatrix = self.subset_data(self.args.betaValueMatrix, self.args.CpGs)
        
        self.labels = self.init_labels(self.betaMatrix)

    def subset_data(self, betaValueMatrix, CpGs):
        CpGs_list = pd.read_csv(CpGs, header=None).iloc[:,0].to_numpy()
        raw_data = pd.read_csv(betaValueMatrix, sep=",", low_memory=True, index_col=0)
        CpGs_list_clean = [CpGs for CpGs in CpGs_list if CpGs in list(raw_data.index)]
        if len(CpGs_list) != len(CpGs_list_clean):
            raise ValueError(f"Missing CpGs {set(CpGs_list).difference(set(CpGs_list_clean))}")
        subset_data = raw_data.loc[CpGs_list_clean]
        return subset_data.transpose()


    def init_labels(self, betaValueMatrix):
        if self.args.samplesLabel:
            data_labels = pd.read_csv(self.args.samplesLabel, sep="\t", header=0, index_col=0).squeeze()
        else:
            data_labels = pd.DataFrame({"LABEL": "to_predict"} , index=betaValueMatrix.index).rename_axis('SAMPLE_ID').squeeze()
        return data_labels


    def UMAP_reduction(self, data, dimensions=10):
        reducer = umap.UMAP(n_components = dimensions)
        reducer.fit(data)
        
        embedding = pd.DataFrame(reducer.transform(data))
        return reducer, embedding

    @staticmethod
    def split_data(random_seed, x, y):
        x_train, x_test, y_train, y_test  = train_test_split(x, y, test_size=0.20, random_state=random_seed, stratify=y)
        print(f"dimension train dataset: {x_train.shape}")
        print(f"dimension validation dataset: {x_test.shape}")
        return x_train, x_test, y_train, y_test


    @staticmethod
    def build_rf(x, y):
        # build RF model
        model_rf = RandomForestClassifier(n_estimators=100,
                                            criterion='gini',
                                            max_depth=None,
                                            min_samples_split=2,
                                            min_samples_leaf=1,
                                            min_weight_fraction_leaf=0.0,
                                            max_features='sqrt',
                                            max_leaf_nodes=None,
                                            min_impurity_decrease=0.0,
                                            bootstrap=True,
                                            oob_score=False,
                                            n_jobs=None,
                                            random_state=None,
                                            verbose=0,
                                            warm_start=False,
                                            class_weight=None,
                                            ccp_alpha=0.0,
                                            max_samples=None,)
        # fit the model
        model_rf.fit(x, y)
        return model_rf

    @staticmethod
    def build_knn(x, y):
        # build KNN model
        model_knn = KNeighborsClassifier(10, weights="distance" )
        # fit the model
        model_knn.fit(x, y)
        return model_knn


    @staticmethod
    def compute_confidence(df):
        if (df['RF_classification_proba'] > 0.75) & (df['KNN_classification_proba'] > 0.75):
            return "high"
        elif (df['RF_classification_proba'] > 0.75) & (df['KNN_classification_proba'] > 0.25) | (df['RF_classification_proba'] > 0.25) & (df['KNN_classification_proba'] > 0.75):
            return "moderate"
        else:
            return "low"

    @staticmethod
    def all_proba_with_model(model, x, y):
        return pd.DataFrame(model.predict_proba(x),columns= model.classes_, index=y.index)

    @staticmethod
    def select_best_prediction(proba, y):
        return pd.concat([pd.Series(y,name="expected_labels"),
                        pd.Series(proba.idxmax(axis=1),name="predicted_labels", index=y.index),
                        pd.Series(proba.max(axis=1),name="classification_proba", index=y.index)], axis=1)


    def create_final_report(self,expected_y, res_model_rf, res_model_knn):      
        final_res = pd.concat([pd.Series(expected_y,name="expected_labels"),
                        res_model_rf[["predicted_labels", "classification_proba"]].add_prefix('RF_'),
                        res_model_knn[["predicted_labels", "classification_proba"]].add_prefix('KNN_')], 
                        axis=1)
        final_res["confidence"] = final_res.apply(self.compute_confidence, axis = 1) 
        return final_res
    

    def save_pickle_model(self, output_file, model):
        if not os.path.exists(self.args.outClassifier):
            os.makedirs(self.args.outClassifier)
        with open(os.path.join(self.args.outClassifier, output_file), 'wb') as data_fit:
            pickle.dump(model, data_fit)


    def create_classifier(self):
        model_umap, reduced_data = self.UMAP_reduction(self.betaMatrix)
        print("-- Create UMAP")
        # Monte Carlo cross validadation
        rf_classifers=[]
        knn_classifier=[]

        for it,CV_rep in enumerate(range(40,50)):
            print(f"\n-- Create classifier, CV={it}")
            x_train, x_test, y_train, y_test = self.split_data(CV_rep, reduced_data, self.labels)
            # build RF model
            model_rf = self.build_rf(x_train, y_train)
            # predict the test set results
            predictions_rf_test = model_rf.predict(x_test)
            print(f"RF: Prencentage of well classified samples ({len(predictions_rf_test)} samples in the dataset):")
            print(f"{accuracy_score(y_test, predictions_rf_test)*100} %")
            rf_classifers.append(model_rf)
            
            # build KNN model
            model_knn = self.build_knn(x_train, y_train)
            # predict the test set results
            predictions_knn_test = model_knn.predict(x_test)
            print(f"KNN: Prencentage of well classified samples ({len(predictions_knn_test)} samples in the dataset):")
            print(f"{accuracy_score(y_test, predictions_knn_test)*100} %")
            knn_classifier.append(model_knn)     

        if self.args.saveDetailsReport:
            rf_all_proba = self.all_proba_with_model(model_rf, x_test, y_test)
            rf_res = self.select_best_prediction(rf_all_proba, y_test)
            knn_all_proba = self.all_proba_with_model(model_knn, x_test, y_test)
            knn_res = self.select_best_prediction(knn_all_proba, y_test)
            self.create_final_report(y_test, rf_res, knn_res).to_csv(self.args.saveDetailsReport,sep="\t")
        
        # save models    
        self.save_pickle_model("umap_model.pickle", model_umap)
        self.save_pickle_model("rf_models.pickle", rf_classifers)
        self.save_pickle_model("knn_models.pickle", knn_classifier)

    
    def create_prediction(self):
        with open(os.path.join(self.args.classifier,'umap_model.pickle'), 'rb') as umap_fit:
            reducer = pickle.load(umap_fit)
        x_predict = pd.DataFrame(reducer.transform(self.betaMatrix))

        with open(os.path.join(self.args.classifier,'rf_models.pickle'), 'rb') as rf_fit:
            list_rf_models = pickle.load(rf_fit)

        with open(os.path.join(self.args.classifier,'knn_models.pickle'), 'rb') as knn_fit:
            list_knn_models = pickle.load(knn_fit)

        rf_predict_CV=[]
        knn_predict_CV=[]
        for model_rf, model_knn in zip(list_rf_models,list_knn_models):
            rf_all_proba = self.all_proba_with_model(model_rf, x_predict,  self.labels)
            rf_predict_CV.append(rf_all_proba)
            knn_all_proba = self.all_proba_with_model(model_knn, x_predict,  self.labels)
            knn_predict_CV.append(knn_all_proba)

        rf_final = self.select_best_prediction(sum(rf_predict_CV)/10, self.labels)
        knn_final = self.select_best_prediction(sum(knn_predict_CV)/10, self.labels)

        self.create_final_report(self.labels, rf_final, knn_final).to_csv(self.args.outPrediction,sep="\t")
    

    def get_args(self):
        parser = ArgumentParser(description="Methylation Classifier")
        subs = parser.add_subparsers(dest="command")

        # build
        build_parser = subs.add_parser("build", help="Build classifier")
        build_parser.add_argument("--CpGs", help="File with CpGs to keep (one ID by row)", required=True)
        build_parser.add_argument("--betaValueMatrix", help="Input matrix of BetaValue", required=True)
        build_parser.add_argument("--samplesLabel", help="Correspondance between ID samples and classification labels", required=True)
        build_parser.add_argument("--saveDetailsReport", help="output details of one cross validation", default='')
        build_parser.add_argument("--outClassifier", help="Output dir to save the classifer", default="./Classifier/")

        # prediction
        predict_parser = subs.add_parser("predict", help='Predict class of new methylation profiles')
        predict_parser.add_argument("--CpGs", help="File with CpGs to keep (one ID by row)")
        predict_parser.add_argument("--betaValueMatrix", help="Input matrix of BetaValue")
        predict_parser.add_argument("--classifier", help="Input dir of classifer", default="./Classifier/")
        predict_parser.add_argument("--samplesLabel", help="Expected correspondance between ID samples and classification labels (not mandatory)")
        predict_parser.add_argument("--outPrediction", help="Output dir to save the classifer", default="./prediction_report.tsv")

        return parser.parse_args()

    def main(self):
        if self.args.command == "build":
            self.create_classifier()
        elif self.args.command == "predict":            
            self.create_prediction()

if __name__ == "__main__":
    MethylClassifier().main()