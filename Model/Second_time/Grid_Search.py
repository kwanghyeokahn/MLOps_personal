import configparser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import joblib
import os
import time
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, ParameterGrid, learning_curve
from sklearn.metrics import r2_score, roc_curve, auc, RocCurveDisplay, f1_score, confusion_matrix, accuracy_score, precision_score, recall_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import ast
import random 
import warnings
warnings.filterwarnings("ignore")

class ModelGridSearch_Module:
    def __init__(self, config_file=None):
        self.config = configparser.ConfigParser()
        self.config.read(config_file, encoding='utf-8') #confing.ini에 주석이라도 한글 있을시 encoding 필요

        #self.train_x = np.array(pd.read_csv(self.config['MODEL_SECOND']['TRAIN_X']))
        self.train_x = np.load(self.config['MODEL_SECOND']['TRAIN_X'])
        self.train_y = np.array(pd.read_csv(self.config['MODEL_SECOND']['TRAIN_Y'])).flatten()
        #self.test_x = np.array(pd.read_csv(self.config['MODEL_SECOND']['TEST_X']))
        self.test_x = np.load(self.config['MODEL_SECOND']['TEST_X'])
        self.test_y = np.array(pd.read_csv(self.config['MODEL_SECOND']['TEST_Y'])).flatten()
        self.task = self.config['MODEL_SECOND']['TASK']
        self.model = self.config['MODEL_SECOND']['MODEL'] 
        self.param_grid = ast.literal_eval(self.config['GRID_PARAMS'][self.model])

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

        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title + ' - Confusion matrix')
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
            for i in set(results_df_regression['Mean Score'][0].keys()):
                plt.figure(figsize=(10, 5))
                plt.bar(results_df_regression['Model'], [score[str(i)] for score in results_df_regression['Mean Score']], color='skyblue')
                plt.title('Regression Model Selection Results')
                plt.grid(True)
                plt.xlabel('Model')
                plt.ylabel(str(i))
                plt.ylim(0, 1)
                plt.xticks(rotation=45)
                plt.savefig(f'{save_path}/{i}_evauation_each_model', dpi=100, bbox_inches='tight')
                plt.close('all')
        else:
            results_df_classification = pd.DataFrame(model_result)

            # Classification 결과 시각화 (Confusion Matrix)
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
    
    def grid_search_average_time(self, model):
        param_grid = ParameterGrid(self.param_grid)
        # 샘플링할 하이퍼파라미터 조합 수
        num_samples = 10
        if num_samples > len(list(param_grid)):
            num_samples = len(list(param_grid))
        else:
            num_samples = int(len(list(param_grid)) * 0.2)
        # 무작위로 샘플링하여 훈련 시간 측정
        sampled_params = random.sample(list(param_grid), num_samples)
        total_time = 0 + 5
        for params in sampled_params:
            start_time = time.time()
            model.set_params(**params)
            model.fit(self.train_x, self.train_y)
            end_time = time.time()
            elapsed_time = end_time - start_time
            total_time += elapsed_time  
        
        # 평균 시간 계산하여 전체 그리드 예상 소요 시간 계산
        average_time_per_sample = total_time / num_samples
        total_combinations = len(param_grid)
        estimated_total_time = average_time_per_sample * total_combinations 
        hours = int(estimated_total_time // 3600)
        minutes = int((estimated_total_time % 3600) // 60)
        seconds = int(estimated_total_time % 60)              
        print(f"Grid Search 전체 조합 수 : {total_combinations}")
        print(f"Grid Search 예상 평균 소요 시간 : {hours} 시간 {minutes} 분 {seconds} 초")
        return total_time
        
    def model_grid_search(self):
        task = self.task
        
        # 결과를 저장할 폴더 생성
        if task == 'regression':
            if not os.path.exists('./Model/Second_time/Regression'):
                os.makedirs('./Model/Second_time/Regression')
            result_dir = './Model/Second_time/Regression'
        else:
            if not os.path.exists('./Model/Second_time/Classification'):
                os.makedirs('./Model/Second_time/Classification')        
            result_dir = './Model/Second_time/Classification'
            
        if task == 'regression':
            scoring = 'r2'
            if self.model == 'Linear Regression':
                models = [
                    ('Linear Regression', LinearRegression())
                ]
            elif self.model == 'Random Forest Regressor':
                models = [
                    ('Random Forest Regressor', RandomForestRegressor())
                ]
            elif self.model == 'SVR':
                models = [
                    ('SVR', SVR())
                ] 
            elif self.model == 'MLP Regressor':
                models = [
                    ('MLP Regressor', MLPRegressor())
                ]                 
            elif self.model == 'XGBoost Regressor':
                models = [
                    ('XGBoost Regressor', xgb.XGBRegressor())
                ]                 
            else:
                raise ValueError("regression model name error")
        elif task == 'classification':
            scoring = 'f1_weight'
            num_classes = len(set(self.train_y))
            if num_classes == 2:
                # 이진 분류
                if self.model == 'Logistic Regression':
                    models = [
                        ('Logistic Regression', LogisticRegression())
                    ]
                elif self.model == 'Random Forest Classifier':
                    models = [
                        ('Random Forest Classifier', RandomForestClassifier())
                    ]
                elif self.model == 'SVC':
                    models = [
                        ('SVC', SVC())
                    ] 
                elif self.model == 'MLP Classifier':
                    models = [
                        ('MLP Classifier', MLPClassifier())
                    ]                 
                elif self.model == 'XGBoost Classifier':
                    models = [
                        ('XGBoost Classifier', xgb.XGBClassifier())
                    ]                 
                else:
                    raise ValueError("classifier model name error")
                
            else:
                # 다중 분류
                if self.model == 'Logistic Regression':
                    models = [
                        ('Logistic Regression', LogisticRegression())
                    ]
                elif self.model == 'Random Forest Classifier':
                    models = [
                        ('Random Forest Classifier', RandomForestClassifier())
                    ]
                elif self.model == 'SVC':
                    models = [
                        ('SVC', SVC())
                    ] 
                elif self.model == 'MLP Classifier':
                    models = [
                        ('MLP Classifier', MLPClassifier())
                    ]                 
                elif self.model == 'XGBoost Classifier':
                    models = [
                        ('XGBoost Classifier', xgb.XGBClassifier())
                    ]                 
                else:
                    raise ValueError("classifier model name error")
        else:
            raise ValueError("task 매개변수는 'regression' 또는 'classification'이어야 합니다.")

        best_model = None
        best_model_name = None    
        results = {'Model': [], 'Mean Score': [], 'Predictions': [], 'True Values': []}
        for name, model in models:
            # 모델 결과를 저장할 폴더 생성
            each_model_dir = os.path.join(result_dir,f'{name}')
            
            if not os.path.exists(each_model_dir):
                os.makedirs(each_model_dir)
                
            print('======expected_average_elapsed_time_for_training_model_grid_search======')
            # 그리드 서치 예상 평균 수행 시간
            predicted_time = self.grid_search_average_time(model)         
            
            if task == 'regression':  
                # 그리드 서치 객체 생성
                grid_search = GridSearchCV(estimator=model, param_grid=self.param_grid, cv=5, scoring=scoring) # if 문으로 scoring 구뷴하기 !
            elif task == 'classification':
                grid_search = GridSearchCV(estimator=model, param_grid=self.param_grid, cv=5, scoring=scoring) # if 문으로 scoring 구뷴하기 !      
                       
            # 그리드 서치 수행
            print('======training_model_grid_search_processing======')
            grid_search.fit(self.train_x, self.train_y)            
            print('======training_model_grid_search_processing_end======')            
            # 최적의 하이퍼파라미터 출력
            model_params = grid_search.best_params_
            print("최적의 하이퍼파라미터:", model_params)
            # 최적의 모델 저장            
            best_model = grid_search.best_estimator_
            
            ##### ==> scoring ...
            learning_curve_save_path = each_model_dir+f'/{name}_train_learning_curve.png'
            #scoring = 'f1'
            self.plot_learning_curve(best_model, f'{name}', self.train_x, self.train_y, cv=5, scoring=scoring, save_path_png=learning_curve_save_path, img_resolution=100)
            
            joblib.dump(best_model, each_model_dir+f'/{name}_best_model.pkl')            
            # 저장된 모델 불러오기 및 예측
            print('======test_data_prediction_processing======')
            loaded_model = joblib.load(each_model_dir+f'/{name}_best_model.pkl')
            predictions = loaded_model.predict(self.test_x)
            print('======test_data_prediction_processing_end======')
            # 평가 결과
            print('======evaluation_processing======')   
                           
            if task == 'classification':
                if num_classes == 2:
                    accuracy = accuracy_score(self.test_y, predictions)
                    precision = precision_score(self.test_y, predictions, average='weighted')
                    recall = recall_score(self.test_y, predictions, average='weighted')
                    f1 = f1_score(self.test_y, predictions, average='micro')
                    fpr, tpr, _ = roc_curve(self.test_y, predictions)
                    roc_auc = auc(fpr, tpr)
                    #print(f'Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}')
                    mean_score = {'F1-SCORE': f1, 'AUC': roc_auc, 'ACCURACY': accuracy, 'PRECISION': precision, 'RECALL': recall}
                    mean_score_result = {'MODEL': name,'F1-SCORE': f1, 'AUC': roc_auc, 'ACCURACY': accuracy, 'PRECISION': precision, 'RECALL': recall}
                    save_result_calssification = pd.DataFrame(mean_score_result, index=[0])
                    save_result_calssification.to_csv(each_model_dir+f'/{name}_train_result.csv',index=False, encoding='CP949')
                    RocCurveDisplay.from_estimator(loaded_model, self.test_x, self.test_y, ax=plt.gca(), name=f'{name} (AUC = {roc_auc:.2f})')
                else:
                    accuracy = accuracy_score(self.test_y, predictions)
                    precision = precision_score(self.test_y, predictions, average='weighted')
                    recall = recall_score(self.test_y, predictions, average='weighted')
                    f1 = f1_score(self.test_y, predictions, average='weighted')
                    mean_score = {'F1-SCORE': f1, 'ACCURACY': accuracy, 'PRECISION': precision, 'RECALL': recall}
                    mean_score_result = {'MODEL': name,'F1-SCORE': f1, 'ACCURACY': accuracy, 'PRECISION': precision, 'RECALL': recall}
                    save_result_calssification = pd.DataFrame(mean_score_result, index=[0])
                    save_result_calssification.to_csv(each_model_dir+f'/{name}_train_result.csv',index=False, encoding='CP949')                                    
            elif task == 'regression':
                r2 = round(r2_score(self.test_y, predictions),2)
                mae = round(mean_absolute_error(self.test_y, predictions),2)
                mape = round(mean_absolute_percentage_error(self.test_y, predictions),2)
                rmse = round(mean_squared_error(self.test_y, predictions, squared=False),2)
                mean_score = {'R-squared': r2, 'MAE': mae, 'MAPE': mape, 'RMSE': rmse}
                #print(f'{name} - R-squared: {r2:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}, RMSE: {rmse:.2f}')
                # 회귀 모델 학습 정도 시각화
                plt.figure(figsize=(8, 6))
                plt.scatter(self.test_y, predictions, c='blue', alpha=0.6)
                plt.plot([self.test_y.min(), self.test_y.max()], [self.test_y.min(), self.test_y.max()], 'k--', lw=2)
                plt.xlabel('True Values')
                plt.ylabel('Predictions')
                plt.title(f'{name} - True Values vs. Predictions')
                plt.savefig(each_model_dir+f'/{name}_train_result', dpi=100, bbox_inches='tight')  # '그래프.png'에 저장하고 dpi와 bbox_inches 설정
                plt.close('all')
                print(mean_score)
            else:
                raise ValueError("task 매개변수는 'regression' 또는 'classification'이어야 합니다.")
            
            # 결과 저장
            results['Model'].append(name)
            results['Mean Score'].append(mean_score)
            results['Predictions'].append(predictions)
            results['True Values'].append(self.test_y)    
                                            
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
                
        best_params_filename = os.path.join(result_dir, f'best_parameter_model_{task}.txt')
        with open(best_params_filename, 'w') as file:
            file.write(f"[{self.task} 모델 파라미터]\n")
            for key, value in model_params.items():
                file.write(f"{key}: {value}\n")
            print(f"결과가 {result_dir} 파일에 저장되었습니다.")
            
        return predicted_time                    
                    
def main():
    # Config 파일 경로
    config_file = './Config/config.ini'
    # 데이터 전처리 및 스케일링 객체 생성
    pred_model = ModelGridSearch_Module(config_file=config_file)
    start_time = time.time() 
    
    predicted_time = pred_model.model_grid_search()
    
    end_time = time.time() 
    elapsed_time = end_time - start_time - predicted_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    print(f"경과 시간 : {hours} 시간 {minutes} 분 {seconds} 초")
    
if __name__ == "__main__":
    main()                            
        
        