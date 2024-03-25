import configparser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import joblib
import os
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
import xgboost as xgb
from sklearn.metrics import r2_score, roc_curve, auc, RocCurveDisplay, f1_score, confusion_matrix, accuracy_score, precision_score, recall_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

class ModelEvaluator_pred:
    def __init__(self, config_file=None):
        self.config = configparser.ConfigParser()
        self.config.read(config_file, encoding='utf-8') #confing.ini에 주석이라도 한글 있을시 encoding 필요
        
        #self.X = np.array(pd.read_csv(self.config['PREDICTION_TEST']['TEST_X']))
        #self.y = np.array(pd.read_csv(self.config['PREDICTION_TEST']['TEST_Y']))
        
        self.X = np.load(self.config['PREDICTION_TEST']['TEST_X'])
        self.y = np.array(pd.read_csv(self.config['PREDICTION_TEST']['TEST_Y'])).flatten()
        self.result_dir = None
        self.task = self.config['PREDICTION_TEST']['TASK'] 
        self.model = self.config['PREDICTION_TEST']['MODEL']
        
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
    
    
    
    def select_model(self):
        
        task = self.task
        model_path = self.model
        model_name_split = model_path.split('\\')
        select_name = model_name_split[-1]
        
        select_name_sec = select_name.split('.')
        select_name_final = select_name_sec[0]
        name = select_name_final
        
        
        # 결과를 저장할 폴더 생성
        if task == 'regression':
            if not os.path.exists('./Prediction/Regression_test'):
                os.makedirs('./Prediction/Regression_test')
            result_dir = './Prediction/Regression_test'
        
        else:
            if not os.path.exists('./Prediction/Classification_test'):
                os.makedirs('./Prediction/Classification_test')        
            result_dir = './Prediction/Classification_test'
            
        if task == 'regression':
            #models = [
            #    ('Linear Regression', LinearRegression()),
            #    ('Random Forest Regressor', RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=random_num)),
            #    ('SVR', SVR(kernel='rbf', C=1.0)),
            #    ('MLP Regressor', MLPRegressor(hidden_layer_sizes=(30,15,7), activation = 'relu', learning_rate='constant', learning_rate_init=0.01, max_iter=200, random_state=random_num)),
            #    ('XGBoost Regressor', xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=random_num))
            #]
            models = [
                      joblib.load(self.model)
                      ]
            
        elif task == 'classification':
            num_classes = len(set(self.y))
            if num_classes == 2:
                # 이진 분류
                #models = [
                #    ('Logistic Regression', LogisticRegression(C=1.0, random_state=random_num)),
                #    ('Random Forest Classifier', RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=random_num)),
                #    ('SVC', (SVC(kernel='rbf', C=1.0, random_state=random_num))),
                #    ('MLP Classifier', MLPClassifier(hidden_layer_sizes=(30,15,7),activation = 'relu', learning_rate='constant', learning_rate_init=0.01, max_iter=200, random_state=random_num)),
                #    ('XGBoost Classifier', xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=random_num))
                #]
                models = [
                        joblib.load(self.model)
                        ]
            else:
                # 다중 분류
                #models = [
                #    ('Logistic Regression', LogisticRegression(C=1.0, multi_class='ovr', random_state=random_num)),
                #    ('Random Forest Classifier', RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=random_num)),
                #    ('SVC', SVC(kernel='rbf', C=1.0, random_state=random_num)),
                #    ('MLP Classifier', MLPClassifier(hidden_layer_sizes=(30,15,7),activation = 'relu', learning_rate='constant', learning_rate_init=0.01, max_iter=200, random_state=random_num)),
                #    ('XGBoost Classifier', xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=random_num))
                #]
                models = [
                          joblib.load(self.model)
                          ]
        else:
            raise ValueError("task 매개변수는 'regression' 또는 'classification'이어야 합니다.")
        
        results = {'Model': [], 'Mean Score': [], 'Predictions': [], 'True Values': []}
                
        for target_model in models:
            print('target_model===>', target_model)
            predictions = target_model.predict(self.X)
            if task == 'classification':
                if num_classes == 2:
                    accuracy = accuracy_score(self.y, predictions)
                    precision = precision_score(self.y, predictions, average='weighted')
                    recall = recall_score(self.y, predictions, average='weighted')
                    f1 = f1_score(self.y, predictions, average='micro')
                    fpr, tpr, _ = roc_curve(self.y, predictions)
                    roc_auc = auc(fpr, tpr)
                    #print(f'Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}')
                    mean_score = {'F1-SCORE': f1, 'AUC': roc_auc, 'ACCURACY': accuracy, 'PRECISION': precision, 'RECALL': recall}
                    mean_score_result = {'MODEL': name,'F1-SCORE': f1, 'AUC': roc_auc, 'ACCURACY': accuracy, 'PRECISION': precision, 'RECALL': recall}
                    save_result_calssification = pd.DataFrame(mean_score_result, index=[0])
                    save_result_calssification.to_csv(result_dir+f'/{name}_test_result.csv',index=False, encoding='CP949')
                    RocCurveDisplay.from_estimator(target_model, self.X, self.y, ax=plt.gca(), name=f'{name} (AUC = {roc_auc:.2f})')
                else:
                    accuracy = accuracy_score(self.y, predictions)
                    precision = precision_score(self.y, predictions, average='weighted')
                    recall = recall_score(self.y, predictions, average='weighted')
                    f1 = f1_score(self.y, predictions, average='weighted')
                    mean_score = {'F1-SCORE': f1, 'ACCURACY': accuracy, 'PRECISION': precision, 'RECALL': recall}
                    mean_score_result = {'MODEL': name,'F1-SCORE': f1, 'ACCURACY': accuracy, 'PRECISION': precision, 'RECALL': recall}
                    save_result_calssification = pd.DataFrame(mean_score_result, index=[0])
                    save_result_calssification.to_csv(result_dir+f'/{name}_testresult.csv',index=False, encoding='CP949')                    
            
            elif task == 'regression':
                r2 = r2_score(self.y, predictions)
                mae = mean_absolute_error(self.y, predictions)
                mape = mean_absolute_percentage_error(self.y, predictions)
                rmse = mean_squared_error(self.y, predictions, squared=False)
                mean_score = {'R-squared': r2, 'MAE': mae, 'MAPE': mape, 'RMSE': rmse}
                #print(f'{name} - R-squared: {r2:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}, RMSE: {rmse:.2f}')
                # 회귀 모델 학습 정도 시각화
                plt.figure(figsize=(8, 6))
                plt.scatter(self.y, predictions, c='blue', alpha=0.6)
                plt.plot([self.y.min(), self.y.max()], [self.y.min(), self.y.max()], 'k--', lw=2)
                plt.xlabel('True Values')
                plt.ylabel('Predictions')
                plt.title(f'{name} - True Values vs. Predictions')
                plt.savefig(result_dir+f'/{name}_train_result', dpi=100, bbox_inches='tight')  # '그래프.png'에 저장하고 dpi와 bbox_inches 설정
                plt.close('all')
            else:
                raise ValueError("task 매개변수는 'regression' 또는 'classification'이어야 합니다.")                
            
            results['Model'].append(name)
            results['Mean Score'].append(mean_score)
            results['Predictions'].append(predictions)
            results['True Values'].append(self.y)       
             
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
                
        plt.close('all')
        return #best_model_name, results #, best_model
        
def main():
    # Config 파일 경로
    config_file = './Config/config.ini'
    
    current_directory = os.getcwd() 
    print('pwd', current_directory)
    
    # 데이터 전처리 및 스케일링 객체 생성
    pred_model = ModelEvaluator_pred(config_file=config_file)
    pred_model.select_model()
    
if __name__ == "__main__":
    main()    