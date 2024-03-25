import pandas as pd
import numpy as np
import os
import configparser
import joblib
from pyswarm import pso

class pso_create_min_max_set:
    def __init__(self, config_file=None):
        self.config = configparser.ConfigParser()
        self.config.read(config_file, encoding='utf-8') #confing.ini에 주석이라도 한글 있을시 encoding 필요

        self.MIN_MAX_TARGET_DATA_PATH = pd.read_csv(self.config['SOLUTION_PSO']['MIN_MAX_TARGET_DATA_PATH'],encoding='CP949')
        self.POSITIVE_COLUMN_SELECT = self.config['SOLUTION_PSO']['POSITIVE_COLUMN_SELECT']
        self.task  = self.config['SOLUTION_PSO']['TASK']
        
        if self.config['SOLUTION_PSO']['POSITIVE_VALUE'] == 'None':
            self.POSITIVE_VALUE = None
        else :
            self.POSITIVE_VALUE = self.config['SOLUTION_PSO']['POSITIVE_VALUE']
            
        self.categorical_col_list = self.config['SOLUTION_PSO']['CATEGORICAL_COL']
        self.numeric_col_list = self.config['SOLUTION_PSO']['NUMERIC_COL']
        
        self.MODEL = self.config['SOLUTION_PSO']['MODEL'] 
        self.SCALER = self.config['SOLUTION_PSO']['SCALER'] 
        
        self.numeric_col = [value.strip() for value in self.numeric_col_list.split(',')]
        self.categorical_col = [value.strip() for value in self.categorical_col_list.split(',')]
        self.use_col = self.categorical_col + self.numeric_col
        
        
        self.unique_list, self.unique_len = self.dependent_unique_check()
        
        if self.task == 'classification':      
            if self.unique_len == 2:
                dataset_path = './Dataset/PSO/classification/binary_dependent_unique'
                if not os.path.exists(dataset_path):
                    os.makedirs(dataset_path)
                self.dataset_result_dir = dataset_path    
                print('======make_dataset_folder_processing======')

                pso_result_path = './Solution/PSO/classification/result'
                if not os.path.exists(pso_result_path):
                    os.makedirs(pso_result_path)
                self.pso_result_dir = pso_result_path     
                print('======make_pso_result_folder_processing======') 

            elif self.unique_len > 2:
                dataset_path = './Dataset/PSO/classification/multiple_dependent_unique'
                if not os.path.exists(dataset_path):
                    os.makedirs(dataset_path)
                self.dataset_result_dir = dataset_path
                print('======make_dataset_folder_processing======')   

                pso_result_path = './Solution/PSO/classification/result'
                if not os.path.exists(pso_result_path):
                    os.makedirs(pso_result_path)
                self.pso_result_dir = pso_result_path     
                print('======make_pso_result_folder_processing======')                         
            else:
                raise ValueError("CALSSIFICATION CASE의 경우 DEPENDENT_COLUMN_UNIQUE_COUNT는 2개 이상만 가능합니다.")
            
        elif self.task == 'regression':
            
            dataset_path = './Dataset/PSO/regression/dependent_unique'
            if not os.path.exists(dataset_path):
                os.makedirs(dataset_path)
            self.dataset_result_dir = dataset_path    
            print('======make_dataset_folder_processing======')
            pso_result_path = './Solution/PSO/regression/result'
            if not os.path.exists(pso_result_path):
                os.makedirs(pso_result_path)
            self.pso_result_dir = pso_result_path     
            print('======make_pso_result_folder_processing======')     
            
        else:
            raise ValueError("confif.ini에서 SOLUTION_PSO부분의 TASK를 classification 또는 regression 선택해주세요.")        
        
        
        self.SWARMSIZE = int(self.config['SOLUTION_PSO']['SWARMSIZE'])
        self.MAXITER = int(self.config['SOLUTION_PSO']['MAXITER'])
        
    def dependent_unique_check(self):
        dependent_col = self.MIN_MAX_TARGET_DATA_PATH[self.POSITIVE_COLUMN_SELECT]
        unique_list = pd.unique(dependent_col)
        unique_len = len(pd.unique(dependent_col))
        return unique_list, unique_len
    
    def make_min_max_dataset(self):
        save_path_list = []
        
        if self.task == 'classification':
            
            if int(self.unique_len) == 2:
                non_defect_dataset = self.MIN_MAX_TARGET_DATA_PATH[self.MIN_MAX_TARGET_DATA_PATH[self.POSITIVE_COLUMN_SELECT]!=self.POSITIVE_VALUE]
                non_defect_dataset_min = pd.DataFrame(non_defect_dataset.describe().loc['min'].values,columns=['min'])
                non_defect_dataset_max = pd.DataFrame(non_defect_dataset.describe().loc['max'].values,columns=['max'])
                non_defect_dataset_col = pd.DataFrame(non_defect_dataset.columns.values,columns=['features'])
                non_defect_min_max = pd.concat([non_defect_dataset_col,non_defect_dataset_min],axis=1)
                non_defect_min_max = pd.concat([non_defect_min_max,non_defect_dataset_max],axis=1)
                non_defect_min_max = non_defect_min_max.iloc[:-1] #label 정보 제거
                save_path = f'{self.dataset_result_dir}/non_defect_min_max.csv'
                non_defect_min_max.to_csv(save_path, index=False, encoding='CP949')
                print('======binary_unique_minmax_dataset_processing======') 
                save_path_list.append(save_path)
            else :
                for i, unique in enumerate(self.unique_list, start=1):
                    non_defect_dataset = self.MIN_MAX_TARGET_DATA_PATH[self.MIN_MAX_TARGET_DATA_PATH[self.POSITIVE_COLUMN_SELECT] == unique]
                    non_defect_dataset_min = pd.DataFrame(non_defect_dataset.describe().loc['min'].values,columns=['min'])
                    non_defect_dataset_max = pd.DataFrame(non_defect_dataset.describe().loc['max'].values,columns=['max'])
                    non_defect_dataset_col = pd.DataFrame(non_defect_dataset.columns.values,columns=['features'])
                    non_defect_min_max = pd.concat([non_defect_dataset_col,non_defect_dataset_min],axis=1)
                    non_defect_min_max = pd.concat([non_defect_min_max,non_defect_dataset_max],axis=1)
                    non_defect_min_max = non_defect_min_max.iloc[:-1] #label 정보 제거
                    save_path = f'{self.dataset_result_dir}/unique_num_{unique}_min_max.csv'
                    non_defect_min_max.to_csv(save_path, index=False, encoding='CP949')
                    print(f'======multiple_unique_{unique}_minmax_dataset_processing======') 
                    save_path_list.append(save_path)  
                    
        elif self.task == 'regression':
            #non_defect_dataset = self.MIN_MAX_TARGET_DATA_PATH[self.MIN_MAX_TARGET_DATA_PATH[self.POSITIVE_COLUMN_SELECT] == unique]
            non_defect_dataset = self.MIN_MAX_TARGET_DATA_PATH.drop(self.POSITIVE_COLUMN_SELECT, axis=1)
            non_defect_dataset_min = pd.DataFrame(non_defect_dataset.describe().loc['min'].values,columns=['min'])
            non_defect_dataset_max = pd.DataFrame(non_defect_dataset.describe().loc['max'].values,columns=['max'])
            non_defect_dataset_col = pd.DataFrame(non_defect_dataset.columns.values,columns=['features'])
            non_defect_min_max = pd.concat([non_defect_dataset_col,non_defect_dataset_min],axis=1)
            non_defect_min_max = pd.concat([non_defect_min_max,non_defect_dataset_max],axis=1)
            #non_defect_min_max = non_defect_min_max.iloc[:-1] #label 정보 제거            
            save_path = f'{self.dataset_result_dir}/regression_min_max.csv'
            non_defect_min_max.to_csv(save_path, index=False, encoding='CP949')
            print(f'======regression_dataset_minmax_dataset_processing======') 
            save_path_list.append(save_path)  


        return save_path_list
    
    # 상한 PSO 모델
    # 목적식    
    def ub_objective_function(self, x):
        model = joblib.load(self.MODEL)
        loaded_scaler = joblib.load(self.SCALER)        
        X_train_col = self.use_col  

        X_test_df = pd.DataFrame(x).T
        X_test_df.columns = X_train_col
        X_test_df['Charge'] = np.round(X_test_df['Charge'])

        X_test_scaled = loaded_scaler.named_steps['preprocessor'].transform(X_test_df).toarray()
        prod = model.predict(X_test_scaled)        
        
        print(prod)
        
        return -prod # PSO는 기본적으로 목적식의 최소를 찾아가기에 상한을 찾기위해서 음수를 붙임
    
    
    
    # 하한 PSO 모델
    # 목적식            
    def lb_objective_function(self, x):
        model = joblib.load(self.MODEL)
        loaded_scaler = joblib.load(self.SCALER)        
        X_train_col = self.use_col        
        
        X_test_df = pd.DataFrame(x).T
        X_test_df.columns = X_train_col
        X_test_df['Charge'] = np.round(X_test_df['Charge'])
        
        X_test_scaled = loaded_scaler.named_steps['preprocessor'].transform(X_test_df).toarray()
        prod = model.predict(X_test_scaled)   
        return prod # PSO는 기본적으로 목적식의 최소를 찾아가기에 상한을 찾기위해서 양수를 붙임

    # 제약식
    def holding_temp_ub(self, x):
        
        x_org = x
        #print('x_org===>',x_org)
        holding_temp = x_org[3].item()
        
        holding_temp_ub = 1325
        
        return holding_temp_ub - holding_temp
    
    def holding_temp_lb(self, x):
        
        x_org = x
        #print('x_org===>',x_org)
        holding_temp = x_org[3].item()
        
        holding_temp_lb = 1302 
        #보고서 상과 실제 양품데이터 간의 데이터 정보 상이 (실데이터 기준 1303도에서도 양품 나옴)
        #1315
        
        return holding_temp - holding_temp_lb
    
    def after_cooler_temp_ub(self, x):
        
        x_org = x
        #print('x_org===>',x_org)
        after_cooler_temp = x_org[7].item()
        
        after_cooler_temp_ub = 70
        
        return after_cooler_temp_ub - after_cooler_temp    
    
    def after_cooler_temp_lb(self, x):
        
        x_org = x
        #print('x_org===>',x_org)
        after_cooler_temp = x_org[7].item()
        
        after_cooler_temp_lb = 37
        #보고서 상과 실제 양품데이터 간의 데이터 정보 상이 (실데이터 기준 38도에서도 양품 나옴)
        #60
        
        return after_cooler_temp - after_cooler_temp_lb    
    
    
    def waiting_time_ub(self, x):
        
        x_org = x
        #print('x_org===>',x_org)
        waiting_time = x_org[8].item()
        
        waiting_time_ub = 6.1 #5.5
        
        return waiting_time_ub - waiting_time 

    def waiting_time_lb(self, x):
        
        x_org = x
        #print('x_org===>',x_org)
        waiting_time = x_org[8].item()
        
        waiting_time_lb = 4.7 #5.5
        
        return waiting_time - waiting_time_lb 
                                                
                
    def run_pso(self):
        save_path_list = self.make_min_max_dataset()
        
        constraint_function_list = [self.holding_temp_ub, self.holding_temp_lb, self.after_cooler_temp_ub, self.after_cooler_temp_lb, self.waiting_time_ub, self.waiting_time_lb]
        
        if self.task == 'classification': 
        
            for minmax_dataset, self.unique_i in zip(save_path_list, self.unique_list):
                print('minmax_dataset, unique_i ===> ',minmax_dataset, self.unique_i)
                lb_ub_df = pd.read_csv(f'{minmax_dataset}',encoding='CP949')
                lb = lb_ub_df['min'].to_list()
                ub = lb_ub_df['max'].to_list()
                print('ub & lb ==>',ub,lb)

                #constraint_function_list = [self.constraint_function]

                print('======ub_pso_processing======') 
                # ub pso
                ub_xopt, ub_fopt = pso(self.ub_objective_function, lb, ub, 
                                 ieqcons = constraint_function_list, 
                                 swarmsize = self.SWARMSIZE,
                                 maxiter = self.MAXITER, 
                                 debug = True) 
                xopt_df_ub = pd.DataFrame([ub_xopt],columns=lb_ub_df['features'].to_list())
                print('======ub_pso_processing_finish======')

                print('======lb_pso_processing======')             
                # lb pso
                lb_xopt, lb_fopt = pso(self.lb_objective_function, lb, ub, 
                                 ieqcons = constraint_function_list, 
                                 swarmsize = self.SWARMSIZE,
                                 maxiter = self.MAXITER, 
                                 debug = True)             
                xopt_df_lb = pd.DataFrame([lb_xopt],columns=lb_ub_df['features'].to_list())            
                print('======lb_pso_processing_finish======')            

                result_ub_lb = pd.concat([xopt_df_ub,xopt_df_lb])    
                dependent_unique_num_path = f'{self.pso_result_dir}/unique_number_{self.unique_i}'
                if not os.path.exists(dependent_unique_num_path):
                    os.makedirs(dependent_unique_num_path)
                dependent_unique_save_path = f'{dependent_unique_num_path}/{self.unique_i}_PSO_ub_lb_result.csv'
                result_ub_lb.to_csv(dependent_unique_save_path, index=False, encoding='CP949') 

                result_T = result_ub_lb.T
                result_T = result_T.reset_index()
                result_T.columns = ['features','pso_max','pso_min']

                new_order = ['features','pso_min','pso_max']
                result_T = result_T[new_order]
                compare = pd.merge(lb_ub_df, result_T, on='features',how='outer')
                summary_save_path = f'{dependent_unique_num_path}/{self.unique_i}_PSO_ub_lb_compare_result.csv'
                compare.to_csv(summary_save_path, index=False, encoding='CP949')                        
                print('======pso_result_file_making======')   
                
        elif self.task == 'regression':
            
            lb_ub_df = pd.read_csv(f'{save_path_list[0]}',encoding='CP949')
            lb = lb_ub_df['min'].to_list()
            ub = lb_ub_df['max'].to_list()
            #print('------>',lb_ub_df) #상한 값이 무조건 하한보다 초과되야 함!, 동일한 값도 안됨
            
            
            print('======ub_pso_processing======') 
            # ub pso
            ub_xopt, ub_fopt = pso(self.ub_objective_function, lb, ub, 
                             ieqcons = constraint_function_list, 
                             swarmsize = self.SWARMSIZE,
                             maxiter = self.MAXITER, 
                             debug = True) 
            xopt_df_ub = pd.DataFrame([ub_xopt],columns=lb_ub_df['features'].to_list())
            print('======ub_pso_processing_finish======')    
                
            print('======lb_pso_processing======')             
            # lb pso
            lb_xopt, lb_fopt = pso(self.lb_objective_function, lb, ub, 
                             ieqcons = constraint_function_list, 
                             swarmsize = self.SWARMSIZE,
                             maxiter = self.MAXITER, 
                             debug = True)             
            xopt_df_lb = pd.DataFrame([lb_xopt],columns=lb_ub_df['features'].to_list())            
            print('======lb_pso_processing_finish======')   
                  
            result_ub_lb = pd.concat([xopt_df_ub,xopt_df_lb])    
            dependent_unique_num_path = f'{self.pso_result_dir}/regression_pso_dir'
            if not os.path.exists(dependent_unique_num_path):
                os.makedirs(dependent_unique_num_path)
            dependent_unique_save_path = f'{dependent_unique_num_path}/PSO_ub_lb_result.csv'
            result_ub_lb.to_csv(dependent_unique_save_path, index=False, encoding='CP949') 
            result_T = result_ub_lb.T
            result_T = result_T.reset_index()
            result_T.columns = ['features','pso_max','pso_min']
            new_order = ['features','pso_min','pso_max']
            result_T = result_T[new_order]
            compare = pd.merge(lb_ub_df, result_T, on='features',how='outer')
            summary_save_path = f'{dependent_unique_num_path}/PSO_ub_lb_compare_result.csv'
            compare.to_csv(summary_save_path, index=False, encoding='CP949')                        
            print('======pso_result_file_making======')  
                                   
        return
        
    
    

def main():
    # Config 파일 경로
    config_file = './Config/config.ini'
    # 데이터 전처리 및 스케일링 객체 생성
    pred_model = pso_create_min_max_set(config_file=config_file)
    pred_model.make_min_max_dataset()
    pred_model.run_pso()
    
    
if __name__ == "__main__":
    main()    