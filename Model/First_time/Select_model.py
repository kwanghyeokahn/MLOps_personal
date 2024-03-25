import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict, learning_curve
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
import xgboost as xgb
from sklearn.metrics import r2_score, roc_curve, auc, RocCurveDisplay, f1_score, confusion_matrix, accuracy_score, precision_score, recall_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import os
import itertools
import joblib
import configparser
import warnings
warnings.filterwarnings("ignore")




class ModelEvaluator:
    def __init__(self, config_file=None):
        self.config = configparser.ConfigParser()
        self.config.read(config_file, encoding='utf-8') #confing.ini에 주석이라도 한글 있을시 encoding 필요

        #self.X = np.array(pd.read_csv(self.config['MODEL_FIRST']['TRAIN_X']))
        self.X = np.load(self.config['MODEL_FIRST']['TRAIN_X'])
        self.y = np.array(pd.read_csv(self.config['MODEL_FIRST']['TRAIN_Y'])).flatten()
        self.result_dir = None
        self.task = self.config['MODEL_FIRST']['TASK'] 

    def plot_learning_curve(self, estimator, title, X, y, axes=None, ylim=None, cv=None,scoring=None, save_path_png=None,img_resolution=None,
                            n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
        if axes is None:
            _, axes = plt.subplots(1, 3, figsize=(20, 5))
        axes[0].set_title(title)
        if ylim is not None:
            axes[0].set_ylim(*ylim)
        axes[0].set_xlabel("Training examples")
        axes[0].set_ylabel("Score")
        train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
            estimator,
            X,
            y,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            train_sizes=train_sizes,
            return_times=True,
        )
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)
            
                
        # Plot learning curve
        axes[0].grid()
        axes[0].fill_between(
            train_sizes,
            train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std,
            alpha=0.1,
            color="r",
        )
        axes[0].fill_between(
            train_sizes,
            test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std,
            alpha=0.1,
            color="g",
        )
        axes[0].plot(
            train_sizes, train_scores_mean, "o-", color="r", label="Training score"
        )
        axes[0].plot(
            train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
        )
        axes[0].legend(loc="best")
        # Plot n_samples vs fit_times
        axes[1].grid()
        axes[1].plot(train_sizes, fit_times_mean, "o-")
        axes[1].fill_between(
            train_sizes,
            fit_times_mean - fit_times_std,
            fit_times_mean + fit_times_std,
            alpha=0.1,
        )
        axes[1].set_xlabel("Training examples")
        axes[1].set_ylabel("fit_times")
        axes[1].set_title("Scalability of the model")
        # Plot fit_time vs score
        fit_time_argsort = fit_times_mean.argsort()
        fit_time_sorted = fit_times_mean[fit_time_argsort]
        test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
        test_scores_std_sorted = test_scores_std[fit_time_argsort]
        axes[2].grid()
        axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
        axes[2].fill_between(
            fit_time_sorted,
            test_scores_mean_sorted - test_scores_std_sorted,
            test_scores_mean_sorted + test_scores_std_sorted,
            alpha=0.1,
        )
        axes[2].set_xlabel("fit_times")
        axes[2].set_ylabel("Score")
        axes[2].set_title("Performance of the model")
        plt.savefig(save_path_png, dpi=img_resolution, bbox_inches='tight')  # '그래프.png'에 저장하고 dpi와 bbox_inches 설정
        print(f"결과가 {save_path_png} 파일에 저장되었습니다.")
        return    

    def plot_confusion_matrix(self, cm, classes, normalize=False, title=None, cmap=plt.cm.Blues, save=None):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        #print(cm)

        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title+' - Confusion matrix')
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(f'{save}/{title}_evauation_each_model', dpi=100, bbox_inches='tight')
        plt.close('all')

    def visualization_part(self, task_name, model_result, save_path = None):
        if task_name == 'regression':
            # 결과를 데이터프레임으로 변환
            results_df_regression = pd.DataFrame(model_result)

            # 결과 출력
            # Regression 결과 시각화 (예: R-squared)
            #plt.bar(results_df_regression['Model'], [score['R-squared'] for score in results_df_regression['Mean Score']], color='skyblue')
            for i in set(results_df_regression['Mean Score'][0].keys()):
                plt.figure(figsize=(10, 5))
                plt.bar(results_df_regression['Model'], [score[str(i)] for score in results_df_regression['Mean Score']], color='skyblue')
                plt.title('Regression Model Selection Results')
                plt.grid(True)
                plt.xlabel('Model')
                plt.ylabel(str(i))
                plt.ylim(0, 1)
                plt.xticks(rotation=45)
                #plt.show()
                plt.savefig(f'{save_path}/{i}_evauation_each_model', dpi=100, bbox_inches='tight')
                plt.close('all')
        else:
            results_df_classification = pd.DataFrame(model_result)

            # Classification 결과 시각화 (예: Confusion Matrix)
            for i in range(len(results_df_classification)):
                y_true = results_df_classification['True Values'][i]
                y_pred = results_df_classification['Predictions'][i]
                # 다중 클래스 혼동 행렬
                if len(set(y_true)) > 2:
                    cm = confusion_matrix(y_true, y_pred)
                    self.plot_confusion_matrix(cm, classes=list(set(y_true)), title=f'{results_df_classification["Model"][i]}',save=save_path)
                # 이진 분류 혼동 행렬
                elif len(set(y_true)) == 2:
                    cm = confusion_matrix(y_true, y_pred)
                    self.plot_confusion_matrix(cm, classes=list(set(y_true)), title=f'{results_df_classification["Model"][i]}',save=save_path)   
                else :
                    raise ValueError("Value Error")
                
    def prediction_part(self, task_name, model_result, save_path = None): # 평가결과 기준은 회귀 모델 : R2, 분류 모델 : F1 
        if task_name == 'regression':
            # 결과를 데이터프레임으로 변환
            results_df_regression = pd.DataFrame(model_result)
            empty_table = pd.DataFrame()
            for index, (model_name,prediction_value,real_value) in enumerate(zip(results_df_regression['Model'],results_df_regression['Predictions'],results_df_regression['True Values'])):
                empty_table['Predict'] = prediction_value.flatten()
                empty_table['True Values'] = real_value.flatten()
                empty_table.to_csv(f'{save_path}/{model_name}_prediction_value.csv',encoding='CP949',index=False)
                
            for i in set(results_df_regression['Mean Score'][0]):
                results_df_regression[str(i)] = [score[str(i)] for score in results_df_regression['Mean Score']]
            #get_col = ['Model'] + list(set(results_df_regression['Mean Score'][0])) 
            get_col = ['Model'] + list(results_df_regression['Mean Score'][0].keys())
            results_df_summary = results_df_regression[get_col].sort_values(by='R-squared',ascending=False)
            results_df_summary = results_df_summary.reset_index(drop=True)
            results_df_summary.to_csv(f'{save_path}/model_evaluation_summary.csv',encoding='CP949',index=False)
            best_model_name = results_df_summary['Model'][0]
            best_score = results_df_summary['R-squared'][0]
            
        elif task_name == 'classification':
            results_df_classification = pd.DataFrame(model_result)
            empty_table = pd.DataFrame()
            for index, (model_name,prediction_value,real_value) in enumerate(zip(results_df_classification['Model'],results_df_classification['Predictions'],results_df_classification['True Values'])):
                empty_table['Predict'] = prediction_value.flatten()
                empty_table['True Values'] = real_value.flatten()
                empty_table.to_csv(f'{save_path}/{model_name}_prediction_value.csv',encoding='CP949',index=False)
                
            for i in set(results_df_classification['Mean Score'][0]):
                results_df_classification[str(i)] = [score[str(i)] for score in results_df_classification['Mean Score']]
            #get_col = ['Model'] + list(set(results_df_classification['Mean Score'][0]))
            get_col = ['Model'] + list(results_df_classification['Mean Score'][0].keys())
            results_df_summary = results_df_classification[get_col].sort_values(by='F1-SCORE',ascending=False)  
            results_df_summary = results_df_summary.reset_index(drop=True)
            results_df_summary.to_csv(f'{save_path}/model_evaluation_summay.csv',encoding='CP949',index=False)
            
            best_model_name = results_df_summary['Model'][0]
            best_score = results_df_summary['F1-SCORE'][0]
            print(results_df_summary)
    
        else:
            raise ValueError("task name error")
    
        return best_model_name, best_score   
    
    def select_model(self , cv=5):
        
        task = self.task
        
        random_num = 442 #Seed에 대한 민감도가 MLP에서 굉장히 크게 나타남... 향후 GridSearch에서는 Seed 까지도 사용할 필요성 느낌
        
        # 결과를 저장할 폴더 생성

        if task == 'regression':
            if not os.path.exists('./Model/First_time/Regression'):
                os.makedirs('./Model/First_time/Regression')
            result_dir = './Model/First_time/Regression'
        
        else:
            if not os.path.exists('./Model/First_time/Classification'):
                os.makedirs('./Model/First_time/Classification')        
            result_dir = './Model/First_time/Classification'
            
        if task == 'regression':
            scoring = 'r2'
            models = [
                ('Linear Regression', LinearRegression()),
                ('Random Forest Regressor', RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=random_num)),
                #('SVR', SVR(kernel='rbf', C=1.0)),
                ('MLP Regressor', MLPRegressor(hidden_layer_sizes=(30,15,7), activation = 'relu', learning_rate='constant', learning_rate_init=0.01, max_iter=200, random_state=random_num)),
                ('XGBoost Regressor', xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=random_num))
            ]
        elif task == 'classification':
            scoring = 'f1_weighted'
            unique_value = np.unique(self.y)
            num_classes = len(unique_value)
            if num_classes == 2:
                # 이진 분류
                models = [
                    ('Logistic Regression', LogisticRegression(C=1.0, random_state=random_num)),
                    ('Random Forest Classifier', RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=random_num)),
                    ('SVC', (SVC(kernel='rbf', C=1.0, random_state=random_num))),
                    ('MLP Classifier', MLPClassifier(hidden_layer_sizes=(30,15,7),activation = 'relu', learning_rate='constant', learning_rate_init=0.01, max_iter=200, random_state=random_num)),
                    ('XGBoost Classifier', xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=random_num))
                ]
            else:
                # 다중 분류
                models = [
                    ('Logistic Regression', LogisticRegression(C=1.0, multi_class='ovr', random_state=random_num)),
                    ('Random Forest Classifier', RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=random_num)),
                    ('SVC', SVC(kernel='rbf', C=1.0, random_state=random_num)),
                    ('MLP Classifier', MLPClassifier(hidden_layer_sizes=(30,15,7),activation = 'relu', learning_rate='constant', learning_rate_init=0.01, max_iter=200, random_state=random_num)),
                    ('XGBoost Classifier', xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=random_num))
                ]
        else:
            raise ValueError("task 매개변수는 'regression' 또는 'classification'이어야 합니다.")
        
        best_model = None
        best_model_name = None

        # 결과 저장 딕셔너리 초기화
        results = {'Model': [], 'Mean Score': [], 'Predictions': [], 'True Values': []}
        for name, model in models:
            print('model_이름 : ',name)
            # 모델 결과를 저장할 폴더 생성
            each_model_dir = os.path.join(result_dir,f'{name}')
            if not os.path.exists(each_model_dir):
                os.makedirs(each_model_dir)
            
            # 예측 결과와 실제 결과 얻기
            print('X_길이',self.X)
            print('y_길이',self.y)
            
            predictions = cross_val_predict(model, self.X, self.y, cv=cv)
            print(predictions)
            print(predictions[0])
            predictions = np.round(predictions,5)
            
            learning_curve_save_path = each_model_dir+f'/{name}_train_learning_curve.png'
            self.plot_learning_curve(model, f'{name}', self.X, self.y, cv=cv, scoring=scoring, save_path_png=learning_curve_save_path, img_resolution=100)
                 
            model.fit(self.X, self.y)
       
            joblib.dump(model, each_model_dir+f'/{name}_model.pkl')
            
            if task == 'classification':
                if num_classes == 2:
                    accuracy = accuracy_score(self.y, predictions)
                    precision = precision_score(self.y, predictions, average='weighted')
                    recall = recall_score(self.y, predictions, average='weighted')
                    f1 = f1_score(self.y, predictions, average='micro')
                    fpr, tpr, _ = roc_curve(self.y, predictions)
                    roc_auc = auc(fpr, tpr)
                    mean_score = {'F1-SCORE': f1, 'AUC': roc_auc, 'ACCURACY': accuracy, 'PRECISION': precision, 'RECALL': recall}
                    mean_score_result = {'MODEL': name,'F1-SCORE': f1, 'AUC': roc_auc, 'ACCURACY': accuracy, 'PRECISION': precision, 'RECALL': recall}
                    save_result_calssification = pd.DataFrame(mean_score_result, index=[0])
                    save_result_calssification.to_csv(each_model_dir+f'/{name}_train_result.csv',index=False, encoding='CP949')
                    RocCurveDisplay.from_estimator(model, self.X, self.y, ax=plt.gca(), name=f'{name} (AUC = {roc_auc:.2f})')
                else:
                    accuracy = accuracy_score(self.y, predictions)
                    precision = precision_score(self.y, predictions, average='weighted')
                    recall = recall_score(self.y, predictions, average='weighted')
                    f1 = f1_score(self.y, predictions, average='weighted')
                    mean_score = {'F1-SCORE': f1, 'ACCURACY': accuracy, 'PRECISION': precision, 'RECALL': recall}
                    mean_score_result = {'MODEL': name,'F1-SCORE': f1, 'ACCURACY': accuracy, 'PRECISION': precision, 'RECALL': recall}
                    save_result_calssification = pd.DataFrame(mean_score_result, index=[0])
                    save_result_calssification.to_csv(each_model_dir+f'/{name}_train_result.csv',index=False, encoding='CP949')                    
            # 모델 성능 평가 (평균 R-squared 또는 평균 F1 스코어 등)
            elif task == 'regression':
                r2 = round(r2_score(self.y, predictions),2)
                mae = round(mean_absolute_error(self.y, predictions),2)
                mape = round(mean_absolute_percentage_error(self.y, predictions),2)
                rmse = round(mean_squared_error(self.y, predictions, squared=False),2)
                mean_score = {'R-squared': r2, 'MAE': mae, 'MAPE': mape, 'RMSE': rmse}
                # 회귀 모델 학습 정도 시각화
                plt.figure(figsize=(8, 6))
                plt.scatter(self.y, predictions, c='blue', alpha=0.6)
                plt.plot([self.y.min(), self.y.max()], [self.y.min(), self.y.max()], 'k--', lw=2)
                plt.xlabel('True Values')
                plt.ylabel('Predictions')
                plt.title(f'{name} - True Values vs. Predictions')
                plt.savefig(each_model_dir+f'/{name}_train_result', dpi=100, bbox_inches='tight')  # '그래프.png'에 저장하고 dpi와 bbox_inches 설정
                plt.close('all')
            else:
                raise ValueError("task 매개변수는 'regression' 또는 'classification'이어야 합니다.")
            # 결과 저장
            results['Model'].append(name)
            results['Mean Score'].append(mean_score)
            results['Predictions'].append(predictions)
            results['True Values'].append(self.y)

            # 모델 결과를 저장할 폴더 생성
            task_result_dir = os.path.join(result_dir,'Evaluation')
            if not os.path.exists(task_result_dir):
                os.makedirs(task_result_dir)
            self.visualization_part(task, results, task_result_dir)
            best_model_name, best_score = self.prediction_part(task, results, task_result_dir)
            
        # 결과를 파일에 저장
        result_filename = os.path.join(result_dir, f'best_model_{task}.txt')
        with open(result_filename, 'w') as result_file:
            result_file.write(f"Best {task} Model : {best_model_name}\n")
            if task == 'regression':
                result_file.write(f"Best Regression Model Score is R-squared : {best_score:.2f}")
                #result_file.write(f"R-squared: {best_score:.2f}\n")
            else:
                result_file.write(f"Best Classification Model F1 Score: {best_score:.2f}\n")
        return #best_model_name, results #, best_model

def main():

    # Config 파일 경로
    config_file = './Config/config.ini' 
    
    first_model = ModelEvaluator(config_file=config_file)
    first_model.select_model()
    
if __name__ == "__main__":
    main()