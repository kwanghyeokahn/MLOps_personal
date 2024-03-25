import configparser
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

class DataPreprocessingAndScaling:
    def __init__(self, config_file=None):
        self.config = configparser.ConfigParser()
        self.config.read(config_file, encoding='utf-8') #confing.ini에 주석이라도 한글 있을시 encoding 필요

        # Data Preprocessing Configuration
        self.path_data = self.config['DATAPREPROCESSING']['PATH_DATA'] 
        self.Target_col = self.config['DATAPREPROCESSING']['TARGET_COL'] 
        
        self.Categorical_col_list = self.config['DATAPREPROCESSING']['CATEGORICAL_COL']
        self.Numeric_col_list = self.config['DATAPREPROCESSING']['NUMERIC_COL']
        
        self.Positive_value = self.config['DATAPREPROCESSING']['POSITIVE_VALUE'] 
        self.label_method = self.config['DATAPREPROCESSING']['LABEL_METHOD']
        self.outlier_method = self.config['DATAPREPROCESSING']['OUTLIER_METHOD']
        self.save_file_path = self.config['DATAPREPROCESSING']['SAVE_FILE_PATH']
        
        # Data Scaling Configuration
        self.scaler_type = self.config['DATASCALING']['SCALER_TYPE']
        self.scaler_save_path = self.config['DATASCALING']['SCALER_SAVE_PATH']
        # Belong Folder Name
        self.dataset_folder_time = datetime.now().strftime('%Y-%m-%d_%H-%M')
        self.dataset_folder = os.path.join(self.save_file_path, f"Made_{self.dataset_folder_time}") 

    def create_folder(self):
        os.makedirs(self.dataset_folder, exist_ok=True)
        with open(self.dataset_folder+'/Data_info.txt', "w") as file:
            file.write('[데이터 및 전처리에 대한 정보]\n')
        # 1-1. Original_data
        OD_folder = os.path.join(self.dataset_folder, "Original_data")
        os.makedirs(OD_folder, exist_ok=True)
        # 1-2. Remove_outlier_data
        RMOD_folder = os.path.join(self.dataset_folder, "Remove_outlier_data")
        os.makedirs(RMOD_folder, exist_ok=True)
        
        NRMOD_folder = os.path.join(self.dataset_folder, "None_Remove_outlier_data")
        os.makedirs(NRMOD_folder, exist_ok=True)
        # 1-3. Scaler_after_RMoutlier_data
        SARMOD_folder = os.path.join(self.dataset_folder, "Scaler_after_RMoutlier_data")
        os.makedirs(SARMOD_folder, exist_ok=True)
        ## 2. EDA_before_Result : 전처리 전 EDA
        #EDA_before_folder = os.path.join(self.dataset_folder, "EDA_before_data")
        #os.makedirs(EDA_before_folder, exist_ok=True)
        ## 2-1. EDA_After_Result : 전처리 후 EDA
        #EDA_after_folder = os.path.join(self.dataset_folder, "EDA_after_data")
        #os.makedirs(EDA_after_folder, exist_ok=True)        
        
    #def eda(self):
        


    def bring_dataset_labeling(self):
        # Load dataset
        data = pd.read_csv(self.path_data, encoding='CP949')
        # Original_data 파일 저장
        data.to_csv(self.dataset_folder+'/Original_data'+'/Original_data.csv',index=False,encoding='CP949')
        # Data_info.txt 내용 추가
        with open(self.dataset_folder+'/Data_info.txt', "a") as file:
            file.write("원본데이터 파일 이름 : " + f'{self.path_data}' + "\n")
        
        self.Numeric_col = [value.strip() for value in self.Numeric_col_list.split(',')]
        
        self.Categorical_col = [value.strip() for value in self.Categorical_col_list.split(',')]
        use_col = self.Categorical_col + self.Numeric_col
        
        
        
        if self.label_method == 'Classification':  # 지금 버전은 양.불만 구분하는 바라보는 형태 ... 향후 다중 클래스에 대해서 처리 가능하도록 해야해(one-hot encoding 필요! + pkl 필요 !)
            try:
                self.Positive_value = int(self.Positive_value)
                ### 숫자로 들어오는 양품 값에 대해서 0,1로 재정의하는 과정 필요, 추가로 해당 컬럼의 유니크 값이 3이상인 경우는 에러 뱉어줘야해
                label_define = data[self.Target_col].unique()
                if len(label_define) >= 3: # 현재는 one-hot encoding 없이 진행
                    #raise ValueError("Invalid Target count. Only binary label")
                    X = data[use_col].values
                    y = data[self.Target_col].values    
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, shuffle=True, random_state=42)                    
                else:
                    data[self.Target_col] = data[self.Target_col].apply(lambda x: 0 if x == self.Positive_value else 1)
                    X = data[use_col].values
                    y = data[self.Target_col].values    
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, shuffle=True, random_state=42)
            except Exception as e:
                try:
                    data[self.Target_col] = data[self.Target_col].apply(lambda x: 0 if x == self.Positive_value else 1)
                    X = data[use_col].values
                    y = data[self.Target_col].values    
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, shuffle=True, random_state=42)
                except Exception as e:
                    raise ValueError("Invalid Target name. choose correct Target name")
                             
            
        elif self.label_method == 'Regression':  #고민해봐야하는 부분 : 개발자 의도는 Positive value 값이 없어도 동작하는걸 기대하는데, 실제로 동작할지 ... ?
            X = data[use_col].values
            y = data[self.Target_col].values    
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
        else:
            raise ValueError("Invalid value type. Choose from 'Classification' or 'Regression'.")

        X_train_col = use_col   
        y_train_col = self.Target_col
        
        
        return X_train, X_test, y_train, y_test, X_train_col, y_train_col

    def remove_outlier(self, X_train=None, y_train=None, X_train_col=None, y_train_col=None):
           
        if self.outlier_method == 'IQR':
            print('outlier_method===> IQR')
            #train_data = pd.concat([pd.DataFrame(X_train, columns=X_train_col), pd.DataFrame({y_train_col: y_train})], axis=1)
            
            train_data = pd.DataFrame(X_train, columns=X_train_col)  
            y_train_data = pd.DataFrame({y_train_col: y_train})
            column_name = self.Numeric_col #명목 변수는 이상치 제거 하면 안되기에 수치형 변수만을 대상으로 제거
            Q1 = train_data[column_name].quantile(0.25) 
            Q3 = train_data[column_name].quantile(0.75) 
            IQR = Q3 - Q1    
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            filtered_train_data = train_data[(train_data[column_name] >= lower_bound) & (train_data[column_name] <= upper_bound)]
            #filtered_train_data = train_data # 이상치 제거 후 one-hot encoding 불가 (학습 데이터에서 이상치로 데이터가 사라져 테스트 테이터 변경이 안됨 *데이터 부족 원인)
            #outlier row remove
            before_remove_outlier_data_count = len(train_data)
            filtered_train_data = filtered_train_data[column_name].dropna(axis=0)
            
            after_remove_outlier_data_count = len(filtered_train_data)
            
            
            if len(self.Categorical_col) != 0:
                
                #수치형 변수를 기준으로 이상치 제거된 데이터에 명목 변수 병합
                filtered_train_data_merge_cat_col = pd.concat([filtered_train_data, train_data[self.Categorical_col]], axis=1, join='inner') 
                filtered_train_data_merge = pd.concat([filtered_train_data_merge_cat_col, y_train_data], axis=1, join='inner')
                y_train_filtered = filtered_train_data_merge[y_train_col]
                X_train_filtered = filtered_train_data_merge.drop(columns=[y_train_col])                
            
            else:
                filtered_train_data_merge = pd.concat([filtered_train_data, y_train_data], axis=1, join='inner')
                y_train_filtered = filtered_train_data_merge[y_train_col]
                X_train_filtered = filtered_train_data_merge.drop(columns=[y_train_col])
        
        
            # Remove_outlier_data 파일 저장
            X_train_filtered.to_csv(self.dataset_folder+'/Remove_outlier_data'+'/Remove_outlier_X_train_data.csv',index=False,encoding='CP949')
            y_train_filtered.to_csv(self.dataset_folder+'/Remove_outlier_data'+'/Remove_outlier_y_train_data.csv',index=False,encoding='CP949')
            # Data_info.txt 내용 추가
            with open(self.dataset_folder+'/Data_info.txt', "a") as file:
                file.write("Outlier_method_type : " + f'{self.outlier_method}'+"\n")
                file.write("Before_remove_outlier_data_count : " + f'{before_remove_outlier_data_count}개'+"\n")
                file.write("After_remove_outlier_data_count : " + f'{after_remove_outlier_data_count}개'+"\n")
                file.write('Remove_outlier 파일 이름 : Remove_outlier_X_train_data.csv\n')
                file.write('Remove_outlier 파일 이름 : Remove_outlier_y_train_data.csv\n')
        
        
        
        elif self.outlier_method == 'None':
            print('outlier_method===> None')
            
            train_data = pd.DataFrame(X_train, columns=X_train_col)  
            y_train_data = pd.DataFrame({y_train_col: y_train})
            
            X_train_filtered = train_data
            y_train_filtered = y_train_data
            
            before_remove_outlier_data_count = len(train_data)
            
            # None_Remove_outlier_data 파일 저장
            X_train_filtered.to_csv(self.dataset_folder+'/None_Remove_outlier_data'+'/None_Remove_outlier_X_train_data.csv',index=False,encoding='CP949')
            y_train_filtered.to_csv(self.dataset_folder+'/None_Remove_outlier_data'+'/None_Remove_outlier_y_train_data.csv',index=False,encoding='CP949')
            # Data_info.txt 내용 추가
            with open(self.dataset_folder+'/Data_info.txt', "a") as file:
                file.write("Outlier_method_type : " + f'{self.outlier_method}'+"\n")
                file.write("Before_remove_outlier_data_count : " + f'{before_remove_outlier_data_count}개'+"\n")
                #file.write("After_remove_outlier_data_count : " + f'{after_remove_outlier_data_count}개'+"\n")
                file.write('Remove_outlier 파일 이름 : Remove_outlier_X_train_data.csv\n')
                file.write('Remove_outlier 파일 이름 : Remove_outlier_y_train_data.csv\n')
                
        else:
            
            raise ValueError("Invalid value type. Choose 'IQR'.")
        

            
            
        return X_train_filtered, y_train_filtered



    def apply_scaler(self, X_train, X_test, X_train_col):
        if self.scaler_type == 'standard':
            scaler = StandardScaler()
        elif self.scaler_type == 'min_max':
            scaler = MinMaxScaler()
        elif self.scaler_type == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError("Invalid scaler_type. Choose from 'standard', 'min_max', or 'robust'.")
        
        
        # 범주형과 수치형 열을 나누기
        categorical_columns = self.Categorical_col
        numeric_columns = self.Numeric_col
        
        if len(self.Categorical_col) != 0:
            # One-Hot Encoding과 스케일링을 위한 ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('cat', OneHotEncoder(), categorical_columns),
                    ('num', scaler, numeric_columns),
                ],
                remainder='passthrough'
            )
        else :
            # One-Hot Encoding과 스케일링을 위한 ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', scaler, numeric_columns)
                ],
                remainder='passthrough'
            )            
        
        # 전처리를 위한 전체 파이프라인
        pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
        
        # 데이터 변환과 스케일링 # 희소 행렬이 아닌 일반적인 2차원 배열로 변환
        X_train_scaled = pipeline.fit_transform(X_train).toarray()
        
        #######
        total_tranformed_len = len(X_train_scaled[0])
        print('total_tranformed_len',total_tranformed_len)
        numeric_columns_len = len(numeric_columns)
        print('numeric_columns_len',numeric_columns_len)
        categorical_columns_len = total_tranformed_len - numeric_columns_len
        print('categorical_columns_len',categorical_columns_len)


        ######
        
        # 모델 저장
        scaler_filename = self.scaler_save_path + f"{self.dataset_folder_time}_{self.scaler_type}_scaler.pkl"
        joblib.dump(pipeline, scaler_filename)        
        
        # 모델 불러오기
        loaded_pipeline = joblib.load(scaler_filename)
        
        # 데이터 프레임 변환
        X_test_df = pd.DataFrame(X_test, columns=X_train_col)  
        
        # 희소 행렬이 아닌 일반적인 2차원 배열로 변환
        X_test_scaled = loaded_pipeline.named_steps['preprocessor'].transform(X_test_df).toarray()
                
        # one-hot encoding을 거친 경우 파생변수가 생기면서, encoding 전 컬럼만으로 데이터 프레임 생성 어려움
        ##################
        
        ## 역변환을 위한 각 변환기에 대한 역변환 적용
        #inverse_transformed_data = pd.DataFrame(columns=X_train.columns)

        ## 명목 변수에 대한 역변환 결과를 데이터프레임에 추가
        #inverse_transformed_data[categorical_columns] = loaded_pipeline.named_steps['preprocessor'] \
        #    .named_transformers_['cat'].inverse_transform(X_test_scaled[:, :categorical_columns_len])


        ## 수치 변수에 대한 역변환 결과를 데이터프레임에 추가
        #inverse_transformed_data[numeric_columns] = loaded_pipeline.named_steps['preprocessor'] \
        #    .named_transformers_['num'].inverse_transform(X_test_scaled[:, categorical_columns_len:])
        #
        #
        #print('역변환 ====>',inverse_transformed_data)
        
        ##################
        
        # csv 저장을 위한 DataFrame 생성
        #X_train_scaled_df = X_train_scaled
        #X_test_scaled_df = X_test_scaled

        #======    

        ##X_train_scaled = scaler.fit_transform(X_train.values)
        #scaler.fit(X_train.values)
        #X_train_scaled = scaler.transform(X_train.values)
        #X_test_scaled = scaler.transform(X_test)

        ## Save the scaler object to a file (optional)
        #scaler_filename = self.scaler_save_path + f"{self.dataset_folder_time}_{self.scaler_type}_scaler.pkl"
        #joblib.dump(scaler, scaler_filename)
        
        ## csv 저장을 위한 DataFrame 생성
        #X_train_scaled_df = pd.DataFrame(X_train_scaled,columns=X_train_col)
        #X_test_scaled_df = pd.DataFrame(X_test_scaled,columns=X_train_col)
        
        # Scaler_after_RMoutlier_data 파일 저장
        SARMOD_folder = os.path.join(self.dataset_folder, "Scaler_after_RMoutlier_data")
        #X_train_scaled_df.to_csv(self.dataset_folder+'/Scaler_after_RMoutlier_data'+'/Scaler_after_RMoutlier_X_train_data.csv',index=False,encoding='CP949')
        #X_test_scaled_df.to_csv(self.dataset_folder+'/Scaler_after_RMoutlier_data'+'/Scaler_after_RMoutlier_X_test_data.csv',index=False,encoding='CP949')
        np.savetxt(self.dataset_folder+'/Scaler_after_RMoutlier_data'+'/Scaler_after_RMoutlier_X_train_data.csv',X_train_scaled, delimiter = ',')
        np.savetxt(self.dataset_folder+'/Scaler_after_RMoutlier_data'+'/Scaler_after_RMoutlier_X_test_data.csv',X_test_scaled, delimiter = ',')

        # Data_info.txt 내용 추가
        with open(self.dataset_folder+'/Data_info.txt', "a") as file:
            file.write("Outlier_method_type : "+ f'{self.scaler_type}' +"\n")
            file.write("Outlier_save_path : "+ f'{scaler_filename}' +"\n")
            file.write('Remove_outlier 파일 이름 : Scaler_after_RMoutlier_X_train_data.csv\n')
            file.write('Remove_outlier 파일 이름 : Scaler_after_RMoutlier_X_test_data.csv\n')
            file.write('One_hot_encoding 길이 : '+ f'{categorical_columns_len}' +"\n")
        return X_train_scaled, X_test_scaled, scaler #numpy 형태로 return
    
    def final_dataset_save(self, X_train_scaled, y_train_filtered, X_test_scaled, y_test, y_train_col, X_train_col):
        
        print('len(X_train_scaled):',len(X_train_scaled))
        print('len(y_train_filtered):',len(X_train_scaled))
        print('len(X_test_scaled):',len(X_train_scaled))
        print('len(y_test):',len(X_train_scaled))
        
        X_train_col_to_save = pd.DataFrame(X_train_col)
        X_train_col_to_save.to_csv(self.dataset_folder + '/' +'train_col_info.csv',index=False, encoding='CP949')
        
        y_test = pd.DataFrame({y_train_col: y_test})
        #save_list = [X_train_scaled, y_train_filtered, X_test_scaled, y_test]
        #save_name_list = ['X_train','y_train','X_test','y_test']
        save_list = [y_train_filtered, y_test]
        save_name_list = ['y_train','y_test']
                
        for i,j in zip(save_list, save_name_list):
            i.to_csv(self.dataset_folder + '/'+j+'.csv',index=False, encoding='CP949')
            
        save_list_array = [X_train_scaled, X_test_scaled]
        save_name_list_array = ['X_train','X_test']            
        for i,j in zip(save_list_array, save_name_list_array):
            np.save(self.dataset_folder + '/'+j+'.npy', i)
            #np.savetxt(self.dataset_folder + '/'+j+'.csv', i, delimiter="," )
            #i.to_csv(self.dataset_folder + '/'+j+'.csv',index=False, encoding='CP949')        
        
        print('Data Preprocessing Done')
            

def main():
    # Config 파일 경로
    config_file = './Config/config.ini' 
    # 데이터 전처리 및 스케일링 객체 생성
    preprocessor = DataPreprocessingAndScaling(config_file=config_file)
    # 하위 폴더 생성
    create_belong_foler = preprocessor.create_folder()
    X_train, X_test, y_train, y_test, X_train_col, y_train_col = preprocessor.bring_dataset_labeling()
    X_train_filtered, y_train_filtered = preprocessor.remove_outlier(X_train=X_train, y_train=y_train, X_train_col=X_train_col, y_train_col=y_train_col)
    # 스케일링 적용
    X_train_scaled, X_test_scaled, scaler = preprocessor.apply_scaler(X_train_filtered, X_test, X_train_col)
    # 최종학습데이터 생성
    preprocessor.final_dataset_save(X_train_scaled, y_train_filtered, X_test_scaled, y_test, y_train_col, X_train_col)
    
if __name__ == '__main__':
    main()

#고민해봐야하는 부분 1 : 개발자 의도는 Positive value 값이 없어도 동작하는걸 기대하는데, 실제로 동작할지 ... ?
